import sys,os
sys.path.append(os.path.dirname (os.path.dirname (os.path.abspath (__file__))))
import argparse
import os
import numpy as np
import torch as th
import torch.distributed as dist
from einops import rearrange, repeat
from mm_diffusion import dist_util, logger
from mm_diffusion.common import set_seed_logger_random, save_audio, save_img, save_multimodal, delete_pkl
from mm_diffusion.multimodal_dpm_solver_plus import DPM_Solver as multimodal_dpm_solver_plus
from mm_diffusion.dpm_solver_plus import DPM_Solver as singlemodal_dpm_solver_plus
from mm_diffusion.evaluator import eval_multimodal
from mm_diffusion.multimodal_datasets import load_data
from mm_diffusion.multimodal_script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict
)
from mm_diffusion.script_util import (
    image_sr_model_and_diffusion_defaults,
    image_sr_create_model_and_diffusion
)



def load_training_data(args):
    data = load_data(
        data_dir=args.ref_path, 
        batch_size=args.batch_size,
        video_size=args.video_size,
        audio_size=args.audio_size,
        num_workers=args.num_workers,
        video_fps=args.video_fps,
        audio_fps=args.audio_fps
    )
   
    for video_batch, audio_batch in data:
        gt_batch = {"video": video_batch, "audio":audio_batch}
        
        yield gt_batch

def main():
    args = create_argparser().parse_args()
    args.video_size = [int(i) for i in args.video_size.split(',')]
    args.audio_size = [int(i) for i in args.audio_size.split(',')]
    
    dist_util.setup_dist(args.devices)
    logger.configure(args.output_dir)
    args = set_seed_logger_random(args)


    logger.log("creating model and diffusion...")
    multimodal_model, multimodal_diffusion = create_model_and_diffusion(
         **args_to_dict(args, [key for key in model_and_diffusion_defaults().keys()])
    )
    sr_model, sr_diffusion = image_sr_create_model_and_diffusion(
        **args_to_dict(args, [key for key in image_sr_model_and_diffusion_defaults().keys()])
    )

    if os.path.isdir(args.multimodal_model_path):
        multimodal_name_list = [model_name for model_name in os.listdir(args.multimodal_model_path) \
            if (model_name.startswith('model') and model_name.endswith('.pt') and int(model_name.split('.')[0][5:])>= args.skip_steps)]
        multimodal_name_list.sort()
        multimodal_name_list = [os.path.join(args.model_path, model_name) for model_name in multimodal_name_list[::1]]
    else:
        multimodal_name_list = [model_path for model_path in args.multimodal_model_path.split(',')]
        
    logger.log(f"models waiting to be evaluated:{multimodal_name_list}")

    if os.path.exists(args.sr_model_path):
        sr_model.load_state_dict_(
            dist_util.load_state_dict(args.sr_model_path, map_location="cpu"), is_strict=args.is_strict
        )
        sr_model.to(dist_util.dev())
        if args.use_fp16:
            sr_model.convert_to_fp16()
    else:
        sr_model = None

    sr_noise=None
    if os.path.exists(args.load_noise):
        sr_noise = np.load(args.load_noise)
        sr_noise = th.tensor(sr_noise).to(dist_util.dev()).unsqueeze(0)
        sr_noise = repeat(sr_noise, 'b c h w -> (b repeat) c h w', repeat=args.batch_size * args.video_size[0])
        if dist.get_rank()==0:
            logger.log(f"load noise form {args.load_noise}...")
    data = load_training_data(args)
    
    for multimodal_save_path in multimodal_name_list:
        multimodal_model.load_state_dict_(
            dist_util.load_state_dict(multimodal_save_path, map_location="cpu"), is_strict=args.is_strict
        )
        
        multimodal_model.to(dist_util.dev())
        if args.use_fp16:
            multimodal_model.convert_to_fp16()
        multimodal_model.eval()

        logger.log(f"sampling samples for {multimodal_save_path}")
        model_name = multimodal_save_path.split('/')[-1]

        groups= 0
        gt_save_path = os.path.join(args.output_dir, model_name, "gt")
        reconstruct_save_path = os.path.join(args.output_dir, model_name, "reconstruct")
        sr_save_path = os.path.join(args.output_dir, model_name, "sr")
        if dist.get_rank() == 0:
            os.makedirs(gt_save_path, exist_ok=True)
            os.makedirs(reconstruct_save_path, exist_ok=True)
            os.makedirs(sr_save_path, exist_ok=True)

        while groups * args.batch_size *  dist.get_world_size()< args.all_save_num: 
              
            model_kwargs = {}
            batch_data = next(data)
            
            idx = 0
            for video, audio in zip(batch_data["video"], batch_data["audio"]):             
                video = video.permute(0, 2, 3, 1)
                video = ((video + 1) * 127.5).clamp(0, 255).to(th.uint8).numpy()  
                audio = audio.numpy()  
                video_output_path = os.path.join(gt_save_path, f"{args.sample_fn}_samples_{groups}_{dist.get_rank()}_{idx}.mp4")
                save_multimodal(video, audio, video_output_path, args)
                idx += 1    

            model_kwargs["audio"] = batch_data["audio"].clone().to(dist_util.dev())
            

            shape = {"video":(args.batch_size , *args.video_size), \
                    "audio":(args.batch_size , *args.audio_size)
                }
            
            if args.sample_fn == 'dpm_solver':
                # TODO
                print(" Do not support dpm_solver now")
                
            elif args.sample_fn == 'dpm_solver++':
                # TODO
                print(" Do not support dpm_solver++ now")
                
            else:
                sample_fn = (
                    multimodal_diffusion.conditional_p_sample_loop if  args.sample_fn=="ddpm" else multimodal_diffusion.ddim_sample_loop
                )
                
                sample = sample_fn(
                    multimodal_model,
                    shape = shape,
                    use_fp16 = args.use_fp16,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    class_scale=args.classifier_scale
                    
                )

            video = ((sample["video"] + 1) * 127.5).clamp(0, 255).to(th.uint8)
            audio = batch_data["audio"]              
            video = video.permute(0, 1, 3, 4, 2)
            video = video.contiguous()

            all_videos = video.cpu().numpy()
            all_audios = audio.numpy()
            
            idx = 0
            for video, audio in zip(all_videos, all_audios):
                video_output_path = os.path.join(reconstruct_save_path, f"{args.sample_fn}_samples_{groups}_{dist.get_rank()}_{idx}.mp4")
                save_multimodal(video, audio, video_output_path, args)
         
                idx += 1                    
                    
                        
            if sr_model is not None:
                model_kwargs = {'low_res': sample["video"]}
                b,f,c,h,w = sample["video"].shape
                shape = (b*f,c,args.large_size, args.large_size)
                model_kwargs['low_res'] = rearrange(model_kwargs['low_res'], 'b f c h w -> (b f) c h w')
                if sr_noise == None:
                    noise = th.randn((b, c, args.large_size, args.large_size)).to(dist_util.dev())
                    noise = repeat(noise, 'b c h w -> (b repeat) c h w', repeat=f)
                else:
                    noise = sr_noise
    
                if args.sr_sample_fn == 'dpm_solver':
                    # sample_fn = singlemodal_dpm_solver
                    # sr_sample = sample_fn(shape = shape, 
                    #     model_fn = sr_model, 
                    #     steps=args.timestep_respacing, 
                    #     model_kwargs=model_kwargs,
                    #     noise=noise)

                    dpm_solver = singlemodal_dpm_solver_plus(model=sr_model, \
                    alphas_cumprod=th.tensor(sr_diffusion.alphas_cumprod, dtype=th.float32), \
                        predict_x0=False,model_kwargs=model_kwargs,)
                
                    sr_sample = dpm_solver.sample(
                        noise,
                        steps=50,
                        order=2,
                        skip_type="time_uniform",
                        method="multistep",
                    )
                elif args.sr_sample_fn == 'dpm_solver++':
                    dpm_solver = singlemodal_dpm_solver_plus(model=sr_model, \
                    alphas_cumprod=th.tensor(sr_diffusion.alphas_cumprod, dtype=th.float32), \
                        predict_x0=True,model_kwargs=model_kwargs,)
                
                    sr_sample = dpm_solver.sample(
                        noise,
                        steps=50,
                        order=2,
                        skip_type="time_uniform",
                        method="multistep",
                    )
                else:
                    sample_fn = (
                        sr_diffusion.p_sample_loop if  args.sr_sample_fn=="ddpm" else sr_diffusion.ddim_sample_loop
                    )

                    sr_sample = sample_fn(
                        sr_model,
                        shape,
                        clip_denoised=args.clip_denoised,
                        model_kwargs=model_kwargs,
                        noise=noise
                    )
        

                sr_sample = ((sr_sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sr_sample = sr_sample.permute(0, 2, 3, 1)
                sr_sample = rearrange(sr_sample, '(b f) h w c-> b f h w c', b=args.batch_size)
                video_sr_samples = sr_sample.contiguous().cpu().numpy()
                           
                idx = 0
                for video, audio in zip(video_sr_samples, all_audios):
                    sr_output_path = os.path.join(sr_save_path, f"{args.sample_fn}_samples_{groups}_{dist.get_rank()}_{idx}.{args.save_type}")
                    
                    save_multimodal(video, audio, sr_output_path, args)
                    idx += 1                    
                    

            groups += 1

            dist.barrier()

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        ref_path="",
        batch_size=16,
        sample_fn="dpm_solver",
        sr_sample_fn="dpm_solver",
        sr_model_path="",
        multimodal_model_path="",
        output_dir="",
        save_type="mp4",
        classifier_scale=0.0,
        devices=None,
        is_strict=True,
        all_save_num= 1024,
        seed=42,
        video_fps=10,
        audio_fps=16000,
        load_noise="",
        num_workers=4
    )
   
    defaults.update(model_and_diffusion_defaults())
    defaults.update(image_sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
