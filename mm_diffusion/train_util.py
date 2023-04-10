import copy
import numpy
import functools
import os
import glob
import blobfile as bf
import torch as th
import torch.distributed as dist
import wandb
import socket
from moviepy.audio.AudioClip import AudioArrayClip
import numpy as np
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from .dpm_solver_plus import DPM_Solver as singlemodal_DPM_Solver
from .common import save_one_image, save_one_video, save_png, save_video

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0
PRE_TRAIN_MODELS={'64x64_classifier.pt':[],'64x64_diffusion.pt':[],
'128x128_classifier.pt':[],'128x128_diffusion.pt':[],'256x256_diffusion_uncond.pt':[],
'512x512_classifier.pt':[],'512x512_diffusion.pt':[],'128_512_upsampler.pt':[],'64_256_upsampler.pt':[]
}




class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        lr=0,
        t_lr=1e-4,
        train_type=None,
        save_type="png",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        class_cond=False,
        use_db=False,
        sample_fn='ddpm',
        audio_fps=16000,
        num_classes=0,
        save_row=8
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.train_type =train_type
        self.save_type = save_type
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.t_lr = t_lr
        self.audio_fps = audio_fps
       
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.class_cond=class_cond
        self.num_classes=num_classes
        self.save_row = save_row
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()
     
        self.use_db =use_db
        if self.use_db ==True and dist.get_rank()==0:
            wandb.login(key="fe40fd39ce7e5d8a74588e6c74344b2d4cda113d")
            wandb.init(
               project=f"{logger.get_dir().split('/')[-2]}",
               entity="ludanruan",
               notes=socket.gethostname(),
               name=f"{logger.get_dir().split('/')[-1]}",
               job_type="training",
               reinit=True)

        self.sync_cuda = th.cuda.is_available()
        self.sample_fn=sample_fn
       
        self._load_and_sync_parameters()
        # print(f"**********load static Done...")
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth
        )
    

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
            
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]
        self.output_model_stastics()
       
        # print(f"**********dist_util.dev:{dist_util.dev()}")
        

        if th.cuda.is_available():
            self.use_ddp = True
                       
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
            print(f"******DDP sync model done...")
           
       
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def output_model_stastics(self):
        
        num_params_total = sum(p.numel() for p in self.model.parameters())
        num_params_train = 0
        num_params_pre_load = 0
        
        for param_group in self.opt.param_groups:
            if param_group['lr'] >0:
                num_params_train += sum(p.numel() for p in param_group['params'] if p.requires_grad==True)
    
        if hasattr(self, 'pre_load_params'):
            num_params_pre_load=sum(p.numel() for name, p in self.model.named_parameters() if name in self.pre_load_params)

        if num_params_total > 1e6:
            num_params_total /= 1e6
            num_params_train /= 1e6
            num_params_pre_load /= 1e6
            params_total_label = 'M'
        elif num_params_total > 1e3:
            num_params_total /= 1e3
            num_params_train /= 1e3
            num_params_pre_load = 1e3
            params_total_label = 'k'

        logger.log("Total Parameters:{:.2f}{}".format(num_params_total, params_total_label))
        logger.log("Total Training Parameters:{:.2f}{}".format(num_params_train, params_total_label))
        logger.log("Total Loaded Parameters:{:.2f}{}".format(num_params_pre_load, params_total_label))
    
    def _transfer_state_dict(self, state_dict, model_keys):
        state_dict_keys = []

        # transfer key name to fit in Unet 3D
        
        logger.log("Transfer parameter name in loaded state_dict")
        
        for key in list(state_dict.keys()):
            new_key = key
            if key.startswith('out.') and key not in model_keys:
                new_key = key.replace('out.','out.0.')
                state_dict[new_key]= state_dict.pop(key)
            state_dict_keys.append(new_key) 

        for model_key in model_keys:
            if 'Temporal' not in model_key:
                state_dict_key = state_dict_keys.pop(0)
                if 'label_emb' in state_dict_key and 'label_emb' not in model_key:
                    state_dict_key = state_dict_keys.pop(0)     
             
                if self.model.state_dict()[model_key].shape == state_dict[state_dict_key].shape:
                    state_dict[model_key] = state_dict.pop(state_dict_key)
                else:
                    pdb.set_trace()

        return state_dict

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        check_point_name =  resume_checkpoint.split('/')[-1]

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if self.resume_step > 0 and dist.get_rank()==0:
                logger.log(f"continue training from step {self.resume_step}")
            #if dist.get_rank() == 0:
      
           
            state_dict = dist_util.load_state_dict(resume_checkpoint, map_location=dist_util.dev())
  
            if check_point_name in PRE_TRAIN_MODELS and hasattr(self.model, 'frame_num'):
                state_dict = self._transfer_state_dict(state_dict, self.model.state_dict().keys())
            if check_point_name in PRE_TRAIN_MODELS or self.resume_step :
                self.pre_load_params = state_dict.keys()
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict_(
                state_dict
                )

            # self.model.to(dist_util.dev())
        
        #print(dist.get_rank()," ", "before sync model parameters:",sum([i.sum() for i in self.model.parameters()]))
        dist_util.sync_params(self.model.parameters()) 
        #print(dist.get_rank()," ", "after sync model parameters:",sum([i.sum() for i in self.model.parameters()]))
        

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)

        if ema_checkpoint:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                ema_checkpoint, map_location=dist_util.dev()
            )
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)
        #print(dist.get_rank()," ", "before sync ema parameters:",sum([i.sum() for i in ema_params]))
        dist_util.sync_params(ema_params)
        #print(dist.get_rank()," ", "after sync ema parameters:",sum([i.sum() for i in ema_params]))
        
        
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            lr, batch, sr, cond = next(self.data)
            
            loss = self.run_step(batch, cond)
            
            if dist.get_rank() == 0:
                # print(f"loss:{loss.item()}")
                # print(f"lr:{list(set(self.opt.get_lr()))}")
                if self.use_db:
                    wandb_log = { 'loss': loss}
           
            if self.step % self.log_interval == 0:
                if dist.get_rank() == 0 and self.use_db:
                    log = logger.get_current()
                    wandb_log.update({'loss':log.name2val["loss"], 'grad_norm':log.name2val["grad_norm"], 'loss_q0':log.name2val["loss_q0"]})

                logger.dumpkvs()

            if self.step % self.save_interval == 0:
               
                self.save()
                # Run for a finite amount of time in integration tests.
                if 'low_res' in cond.keys() or 'local_cond' in cond.keys():
                    output_path = self.save_sr()
                    
                elif hasattr(self.model, 'video_size'):
                    output_path = self.save_visual() 
                elif hasattr(self.model, 'audio_size'):
                    output_path = self.save_audio() 

                if dist.get_rank() == 0 and self.use_db:
                    if output_path.endswith('gif'):
                        wandb_log={**wandb_log,'sample': wandb.Video(output_path)}
                    elif output_path.endswith('png'):
                        wandb_log={**wandb_log,'sample': wandb.Image(output_path)}
                    
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return

            if dist.get_rank() == 0 and self.use_db:
                wandb.log(wandb_log)
            self.step += 1

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
      
        loss = self.forward_backward(batch, cond) 
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        
        self._anneal_lr()
        self.log_step()
        
        return loss

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()  
        scale = batch.shape[0]/ self.microbatch
        for i in range(0, batch.shape[0], self.microbatch):
            
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(self.microbatch, dist_util.dev())
  
     
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
           
            loss = (losses["loss"] * weights).mean()
            
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
     
            self.mp_trainer.backward(loss/scale)
            return loss

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        # logger.logkv("lr", sum(set(self.opt.get_lr()))/len(set(self.opt.get_lr())))
    
    def save_visual(self):
      
        all_images = []
        all_labels = []
        logger.log("create samples")

        if len(self.ema_params) > 0:
            state_dict=self.mp_trainer.master_params_to_state_dict(self.ema_params[0])
            self.model.load_state_dict(
                state_dict
                )
        group = 0
        while len(all_images)*self.batch_size < self.save_row**2:
            model_kwargs = {}
       
            if self.class_cond:
                classes = th.randint(
                    low=0, high=self.num_classes, size=(1,), device=dist_util.dev()
                ).expand(self.batch_size)
                model_kwargs["y"] = classes
      
            shape =  [self.batch_size, *self.model.video_size]
            save_one = save_one_video
            postfix = "gif"
            
            sample_dict = {'shape':shape,  'model_kwargs':model_kwargs}
      
            if self.sample_fn == 'dpm_solver':
               
                # sample_fn = dpm_solver_sample
                # sample_dict.update({'model_fn': self.model, 'total_N': len(self.diffusion.betas)})
            
                dpm_solver = singlemodal_DPM_Solver(model= self.model, \
                    alphas_cumprod=th.tensor(self.diffusion.alphas_cumprod, dtype=th.float16), \
                        predict_x0=False, model_kwargs=model_kwargs)
                noise = th.randn([self.batch_size, *self.model.video_size]).to(dist_util.dev())
                
                sample_fn = dpm_solver.sample
                sample_dict = {"noise":noise, "steps":50, "order":2, "skip_type":"time_uniform", "method":"multistep"}

                
            else:
                sample_fn = (
            self.diffusion.p_sample_loop if not self.sample_fn == 'ddim' else self.diffusion.ddim_sample_loop
        )
                sample_dict.update({'model':self.model, 'clip_denoised': True})
  
            sample = sample_fn(
                **sample_dict
            )
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            #[4,16,3,64,64]
            
           
            sample = sample.permute(0, 1, 3, 4, 2).contiguous()
            
            dist.barrier()
            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            #[12,16,3,64,64]
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            
            if  dist.get_rank() == 0: 
                logger.log(f"{len(all_images)*self.batch_size} has sampled")
            
            group+=1
        
        output_path =  os.path.join(logger.get_dir(), f"{self.sample_fn}_samples_{self.step}.{postfix}") 
        if dist.get_rank() == 0:  
            if self.save_type == "one":
                all_images = numpy.concatenate(all_images, axis=0)
                save_one(all_images, output_path, row=self.save_row)
            else:
                raise NotImplementedError
        
        logger.log(f"created {len(all_images)} samples")
        dist.barrier()
        
        state_dict = self.mp_trainer.master_params_to_state_dict(self.mp_trainer.master_params)
        self.model.load_state_dict(
                state_dict
                )
        

        return output_path
    
    def save_audio(self):
      
        all_audios = []
        all_labels = []
        logger.log("create samples")

        if len(self.ema_params) > 0:
            state_dict=self.mp_trainer.master_params_to_state_dict(self.ema_params[0])
            self.model.load_state_dict(
                state_dict
                )

        while len(all_audios)*self.batch_size < self.save_row**2:
            model_kwargs = {}
       
            if self.class_cond:
                classes = th.randint(
                    low=0, high=self.num_classes, size=(1,), device=dist_util.dev()
                ).expand(self.batch_size)
                model_kwargs["y"] = classes
      
            shape =  [self.batch_size, *self.model.audio_size]
            sample_dict = {'shape':shape,  'model_kwargs':model_kwargs}
      
            if self.sample_fn == 'dpm_solver':
                

                dpm_solver = singlemodal_DPM_Solver(model= self.model, \
                    alphas_cumprod=th.tensor(self.diffusion.alphas_cumprod, dtype=th.float16), \
                        predict_x0=False, model_kwargs=model_kwargs)
                noise = th.randn([*shape]).to(dist_util.dev())
                
                sample_fn = dpm_solver.sample
                sample_dict = {"noise":noise, "steps":50, "order":2, "skip_type":"time_uniform", "method":"multistep"}
            
            elif self.sample_fn == 'dpm_solver++':
                dpm_solver = singlemodal_DPM_Solver(model=self.model, \
                    alphas_cumprod=th.tensor(self.diffusion.alphas_cumprod, dtype=th.float16), \
                        predict_x0=True,model_kwargs=model_kwargs,)
                
                sample_fn = dpm_solver.sample
                sample_dict = {"noise":noise, "steps":50, "order":2, "skip_type":"time_uniform", "method":"multistep"}

            else:
                sample_fn = (
            self.diffusion.p_sample_loop if not self.sample_fn == 'ddim' else self.diffusion.ddim_sample_loop
        )
                sample_dict.update({'model':self.model, 'clip_denoised': True})
  
            sample_audio = sample_fn(
                **sample_dict
            )
            gathered_sample_audios = [th.zeros_like(sample_audio) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_sample_audios, sample_audio)  # gather not supported with NCCL
            all_audios.extend([sample_audio.cpu().numpy() for sample_audio in gathered_sample_audios])
            if  dist.get_rank() == 0: 
                logger.log(f"{len(all_audios)*self.batch_size} has sampled")
        all_audios = np.concatenate(all_audios, axis=0)
        output_path = os.path.join(logger.get_dir(), f"{self.sample_fn}_samples_steps{self.step}_0.wav")
        if  dist.get_rank() == 0:  

            for  aid,audio in enumerate(all_audios):     
                output_path = os.path.join(logger.get_dir(), f"{self.sample_fn}_samples_steps{self.step}_{aid}.wav")
                          
                audio = audio.T #[len, channel]
                audio = np.repeat(audio, 2, axis=1)
                audio_clip = AudioArrayClip(audio, fps=self.audio_fps)
                audio_clip.write_audiofile(output_path, fps=self.audio_fps)
            
        
        logger.log(f"created {len(all_audios)} samples")
        dist.barrier()

        state_dict = self.mp_trainer.master_params_to_state_dict(self.mp_trainer.master_params)
        self.model.load_state_dict(
                state_dict
                )
        

        return output_path

    def save_sr(self):
      
        all_images = []
        all_labels = []
        logger.log("create samples")

        if len(self.ema_params) > 0:
            state_dict=self.mp_trainer.master_params_to_state_dict(self.ema_params[0])
            self.model.load_state_dict(
                state_dict
                )
        group = 0
        while len(all_images)*self.batch_size < self.save_row**2:
            model_kwargs = {}
            
            lr, hr, sr, cond = next(self.data)
            for key in cond.keys():
                hr = hr.to(dist_util.dev())
                sr = sr.to(dist_util.dev())
                model_kwargs[key] = cond[key].to(dist_util.dev())

            if self.class_cond:
                classes = th.randint(
                    low=0, high=self.num_classes, size=(1,), device=dist_util.dev()
                ).expand(self.batch_size)
                model_kwargs["y"] = classes
        

            if hasattr(self.model,'video_size'):
                shape = [self.batch_size,  *self.model.video_size]
                shape[2] = 3
                save_one = save_one_video
                save_single = save_video
                postfix = "gif"
            else:
                shape = (self.batch_size, 3, self.model.image_size, self.model.image_size)
                save_one = save_one_image
                save_single =save_png
                postfix = "png"

            sample_dict = {'shape':shape,  'model_kwargs':model_kwargs}
            if self.sample_fn == 'dpm_solver':
               
                dpm_solver = singlemodal_DPM_Solver(model= self.model, \
                    alphas_cumprod=th.tensor(self.diffusion.alphas_cumprod, dtype=th.float16), \
                        predict_x0=False, model_kwargs=model_kwargs)
                noise = th.randn([*shape]).to(dist_util.dev())
                
                sample_fn = dpm_solver.sample
                sample_dict = {"noise":noise, "steps":50, "order":2, "skip_type":"time_uniform", "method":"multistep"}
            
            elif self.sample_fn == 'dpm_solver++':
                dpm_solver = singlemodal_DPM_Solver(model=self.model, \
                    alphas_cumprod=th.tensor(self.diffusion.alphas_cumprod, dtype=th.float16), \
                        predict_x0=True,model_kwargs=model_kwargs,)
                
                sample_fn = dpm_solver.sample
                sample_dict = {"noise":noise, "steps":50, "order":2, "skip_type":"time_uniform", "method":"multistep"}

            else:
                sample_fn = (
                    self.diffusion.p_sample_loop if not self.sample_fn == 'ddim' else self.diffusion.ddim_sample_loop
                )
                sample_dict.update({'model':self.model, 'clip_denoised': True})
  
            sample = sample_fn(
                **sample_dict
            )
            sample = th.cat((sr, sample, hr), axis=3)
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)

            if hasattr(self.model,'video_size'):
                sample = sample.permute(0, 1, 3, 4, 2).contiguous()
            else:
                sample = sample.permute(0, 2, 3, 1).contiguous()
            
            for idx,single_sample in enumerate(sample):
                output_path = os.path.join(logger.get_dir(), f"{self.sample_fn}_samples_{group}_{dist.get_rank()}_{idx}_{self.step}.{postfix}") 
                save_single(single_sample.cpu().numpy(), output_path)
            dist.barrier()
            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            #[12,16,3,64,64]
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            if  dist.get_rank() == 0: 
                logger.log(f"{len(all_images)*self.batch_size} has sampled")
            group+=1
        output_path =  os.path.join(logger.get_dir(), f"samples_{self.step}.{postfix}") 
        if  dist.get_rank() == 0:  
            if self.save_type == "one":
                all_images = numpy.concatenate(all_images, axis=0)
                save_one(all_images, output_path, row=self.save_row)
                    
            else:
                raise NotImplementedError
        
        logger.log(f"created {len(all_images)} samples")
        dist.barrier()

        state_dict = self.mp_trainer.master_params_to_state_dict(self.mp_trainer.master_params)
        self.model.load_state_dict(
                state_dict
                )
        

        return output_path

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

            #sample gifs during training


        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()



def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    filename = "model*.pt"
    max_step = 0
    for name in glob.glob(os.path.join(get_blob_logdir(), filename)):
        step = int(name[-9:-3])
        max_step = max(max_step, step)
    if max_step:
        path = bf.join(get_blob_logdir(), f"model{(max_step):06d}.pt")
    
        if bf.exists(path):
            return path
    return None



def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
   
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).

        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
        
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
