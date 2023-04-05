"""
Train a super-resolution model.
"""

import argparse, sys, os
sys.path.append(os.path.dirname (os.path.dirname (os.path.abspath (__file__))))
import torch.nn.functional as F
from mm_diffusion import dist_util, logger
from mm_diffusion.real_image_datasets import load_data
from mm_diffusion.resample import create_named_schedule_sampler
from mm_diffusion.common import set_seed_logger_random
from mm_diffusion.script_util import (
    image_sr_model_and_diffusion_defaults,
    image_sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from mm_diffusion.train_util import TrainLoop

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(args.devices)
    logger.configure(args.output_dir)
    args = set_seed_logger_random(args)

    logger.log("creating model...")
    model, diffusion = image_sr_create_model_and_diffusion(
        **args_to_dict(args, image_sr_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_superres_data(args)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        use_db=args.use_db,
        save_type=args.save_type,
        class_cond=args.sr_class_cond,
        sample_fn=args.sample_fn
    ).run_loop()


def load_superres_data(args):
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.large_size,
        class_cond=args.sr_class_cond,
        num_workers=args.num_workers
    )
    for small_batch, large_batch, sr_batch, model_kwargs in data:
        model_kwargs["low_res"] = small_batch
        yield small_batch, large_batch, sr_batch, model_kwargs



def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        frame_gap=8,
        num_workers=4,
        use_db=False,
        devices="0",
        output_dir="~/tmp",
        seed=42,
        save_type='one',
        sample_fn='dpm_solver'
    )
    defaults.update(image_sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
