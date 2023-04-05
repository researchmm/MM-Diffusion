"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import sys,os
sys.path.append(os.path.dirname (os.path.dirname (os.path.abspath (__file__))))
import argparse
import os
import torch as th
import torch.distributed as dist
from mm_diffusion import dist_util, logger
from mm_diffusion.multimodal_script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict
)
from mm_diffusion.common import set_seed_logger_random, save_audio, save_img, save_multimodal, delete_pkl
from mm_diffusion.evaluator import eval_multimodal
from mm_diffusion.multimodal_datasets import load_data



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
        model_kwargs = {}
        yield gt_batch, model_kwargs

def main():
    args = create_argparser().parse_args()
    args.video_size = [int(i) for i in args.video_size.split(',')]
    args.audio_size = [int(i) for i in args.audio_size.split(',')]
    
    dist_util.setup_dist(args.devices)
    logger.configure(args.output_dir)
    args = set_seed_logger_random(args)
    
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
         **args_to_dict(args, [key for key in model_and_diffusion_defaults().keys()])
    )

    if os.path.isdir(args.model_path):
        model_name_list = [model_name for model_name in os.listdir(args.model_path) \
            if (model_name.startswith('model') and model_name.endswith('.pt') and int(model_name.split('.')[0][5:])>= args.skip_steps)]
        model_name_list.sort()
        model_path_list = [os.path.join(args.model_path, model_name) for model_name in model_name_list[::1]]
    else:
        model_path_list = [model_path for model_path in args.model_path.split(',')]
        
    logger.log(f"models waiting to be evaluated:{model_path_list}")
    data = load_training_data(args)
    for model_path in model_path_list:
        model.load_state_dict_(
            dist_util.load_state_dict(model_path, map_location="cpu"), is_strict=args.is_strict
        )
        
        model.to(dist_util.dev())
        if args.use_fp16:
            model.convert_to_fp16()
        model.eval()

        logger.log(f"conditional sampling samples for {model_path}")
        model_name = model_path.split('/')[-1]
        
        groups= 0
        gt_save_path = os.path.join(args.output_dir, model_name, "gt")
        reconstruct_save_path = os.path.join(args.output_dir, model_name, "reconstract")
        audio_save_path = os.path.join(args.output_dir, model_name, "audio")
        if dist.get_rank() == 0:
            os.makedirs(gt_save_path, exist_ok=True)
            os.makedirs(reconstruct_save_path, exist_ok=True)
            os.makedirs(audio_save_path, exist_ok=True)

        while groups * args.batch_size *  dist.get_world_size() < args.all_save_num: 
        
            
            batch_data, model_kwargs = next(data)
            
            # save gt
            idx = 0
            for video, audio in zip(batch_data["video"], batch_data["audio"]):             
                video = video.permute(0, 2, 3, 1)
                video = ((video + 1) * 127.5).clamp(0, 255).to(th.uint8).numpy()  
                audio = audio.numpy()  
                video_output_path = os.path.join(gt_save_path, f"{args.sample_fn}_samples_{groups}_{dist.get_rank()}_{idx}.mp4")
                save_multimodal(video, audio, video_output_path, args)
                idx += 1        

            model_kwargs["video"] = batch_data["video"].to(dist_util.dev())
           
            shape = {"video":(args.batch_size , *args.video_size), \
                    "audio":(args.batch_size , *args.audio_size)
                }

            if args.sample_fn == 'dpm_solver':
                #TODO
                print("dpm_solver is not implemented yet..")
                

            elif args.sample_fn == 'dpm_solver++':
                #TODO
                print("dpm_solver++ is not implemented yet..")
                

            else:
                sample_fn = (
                    diffusion.conditional_p_sample_loop if  args.sample_fn=="ddpm" else diffusion.ddim_sample_loop
                )

                sample = sample_fn(
                    model,
                    shape=shape,
                    use_fp16 = args.use_fp16,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    class_scale=args.classifier_scale
                    
                )

            video = ((sample["video"] + 1) * 127.5).clamp(0, 255).to(th.uint8)
            audio = sample["audio"]              
            video = video.permute(0, 1, 3, 4, 2)
            video = video.contiguous()

            all_videos = video.cpu().numpy()
            all_audios = audio.cpu().numpy()

               
            idx = 0
            for video, audio in zip(all_videos, all_audios):
                video_output_path = os.path.join(reconstruct_save_path, f"{args.sample_fn}_samples_{groups}_{dist.get_rank()}_{idx}.mp4")
                audio_output_path = os.path.join(audio_save_path, f"{args.sample_fn}_samples_{groups}_{dist.get_rank()}_{idx}.wav")
                
                save_multimodal(video, audio, video_output_path, args)
                save_audio(audio, audio_output_path, args.audio_fps)
       
                idx += 1        
                
            groups += 1
            dist.barrier()

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,

        batch_size=16,
        sample_fn="ddpm",
        model_path="",
        output_dir="",
        save_type="mp4",
        classifier_scale=0.0,
        devices=None,
        is_strict=True,
        all_save_num= 1024,
        seed=42,
        video_fps=10,
        audio_fps=16000,
        ref_path = "",
        num_workers=4
    )
   
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
