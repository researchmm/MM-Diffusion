"""
This code is extended from guided_diffusion: https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/scripts_util.py
"""


import argparse
from einops import rearrange
from . import multimodal_gaussian_diffusion as gd
from .multimodal_respace import SpacedDiffusion, space_timesteps
from .multimodal_unet import MultimodalUNet

def diffusion_defaults():
    """
    Defaults for multi-modal training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )


def model_defaults():
    """
    Defaults for multi-modal training.
    """
    res = dict(
        video_size="16,3,64,64",
        audio_size="1,25600",
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        cross_attention_resolutions="2,4,8",
        cross_attention_windows="1,4,8",
        cross_attention_shift=True,
        video_attention_resolutions="2,4,8",
        audio_attention_resolutions="-1",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        video_type="2d+1d",
        audio_type="1d",
    )
    return res

def model_and_diffusion_defaults():
    res = model_defaults()
    res.update(diffusion_defaults())
    return res

def create_model_and_diffusion(
    video_size,
    audio_size,
    learn_sigma,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    cross_attention_resolutions,
    cross_attention_windows,
    cross_attention_shift,
    video_attention_resolutions,
    audio_attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    video_type="2d+1d",
    audio_type="1d",
    class_cond=False
):
    model = create_model(
        video_size=video_size,
        audio_size=audio_size,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        cross_attention_resolutions=cross_attention_resolutions,
        cross_attention_windows=cross_attention_windows,
        cross_attention_shift=cross_attention_shift,
        video_attention_resolutions=video_attention_resolutions,
        audio_attention_resolutions=audio_attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        video_type=video_type,
        audio_type=audio_type,
       
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def create_model(
    video_size,
    audio_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    cross_attention_resolutions="2,4,8",
    video_attention_resolutions="2,4,8",
    audio_attention_resolutions="2,4,8",
    cross_attention_windows="1,4,8",
    cross_attention_shift=True,
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    use_fp16=False,
    video_type="2d+1d",
    audio_type="1d",
    resblock_updown=True
):
    
    image_size = video_size[-1] 
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    cross_attention_resolutions = [int(i) for i in cross_attention_resolutions.split(',')]
    video_attention_resolutions = [int(i) for i in video_attention_resolutions.split(',')]
    audio_attention_resolutions = [int(i) for i in audio_attention_resolutions.split(',')]
    cross_attention_windows = [int(i) for i in cross_attention_windows.split(',')]

    return MultimodalUNet(
        video_size=video_size,
        audio_size=audio_size,
        model_channels=num_channels,
        video_out_channels=(3 if not learn_sigma else 6),
        audio_out_channels=(1 if not learn_sigma else 2),
        num_res_blocks=num_res_blocks,
        cross_attention_resolutions=cross_attention_resolutions,
        cross_attention_windows=cross_attention_windows,
        cross_attention_shift=cross_attention_shift,
        video_attention_resolutions=video_attention_resolutions,
        audio_attention_resolutions=audio_attention_resolutions,
        video_type=video_type,
        audio_type= audio_type,

        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=None,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
