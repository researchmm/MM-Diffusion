import os
import random
import torch.distributed as dist
import torch as th
import numpy as np
import glob
from PIL import Image
from einops import rearrange
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.audio.AudioClip import AudioArrayClip
from . import logger

# def sample_fn_decord(sample_fn):
#     sample_fn = sample_fn.replace('dpm_solver++_', '')
#     predict_x0, order, method = sample_fn.split('_')
#     predict_x0 = True if predict_x0 == "True" else False
#     order = int(order)
#     return predict_x0, order, method
    
def delete_pkl(fake_dir):
    fake_paths =  glob.glob(os.path.join(fake_dir, "*.pkl"))
    for fake_path in fake_paths:
        os.remove(fake_path)
        print(f"detete pkl from {fake_path}")
    return 


def save_audio(audio, output_path, audio_fps):
    audio = audio.T #[len, channel]
    audio = np.repeat(audio, 2, axis=1)
    audio_clip = AudioArrayClip(audio, fps=audio_fps)
    audio_clip.write_audiofile(output_path, fps=audio_fps)
    return
    
def save_img(video, output_path):
    os.makedirs(output_path, exist_ok=True)
    for idx, img in enumerate(video):
        img_path = os.path.join(output_path, f"{idx:0>8d}.png")
        Image.fromarray(img).convert('RGB').save(img_path)
    return

def save_png(img, output_path):
    Image.fromarray(img).convert('RGB').save(output_path)
    return

def save_multimodal(video, audio, output_path, args):
    imgs = [img for img in video]
    audio = audio.T #[len, channel]
    audio = np.repeat(audio, 2, axis=1)
    audio_clip = AudioArrayClip(audio, fps=args.audio_fps)
    video_clip = ImageSequenceClip(imgs, fps=args.video_fps)
    video_clip = video_clip.set_audio(audio_clip)   
    video_clip.write_videofile(output_path, args.video_fps, audio=True, audio_fps=args.audio_fps)
    return

def save_one_image(images, save_path, row=5):
    images = images[:row**2, ...]
    assert images.shape[0] % row == 0
    images = np.pad(images,((0,0),(2,2),(2,2),(0,0)))
    images = rearrange(images, '(i j) h w c -> (i h) (j w) c', i = row)
    Image.fromarray(images).convert('RGB').save(save_path)
    return True
    
def save_one_video(videos, save_path, row=5):
    videos = videos[:row**2,...]
    assert videos.shape[0] % row == 0
    videos = np.pad(videos,((0,0),(0,0),(2,2),(2,2),(0,0)))
    videos = rearrange(videos, '(i j) f h w c ->  f (i h) (j w) c', i = row)
    imgs = [Image.fromarray(img) for img in videos]
    imgs[0].save(save_path, save_all=True, append_images=imgs[1:], duration=100, loop=0)
    return True

def save_video(result, output_path, video_fps=10):
    ext = (output_path.split('.')[-1]).lower()
    if ext == 'gif':
        imgs = [Image.fromarray(img) for img in result]
        imgs[0].save(output_path, save_all=True, append_images=imgs[1:], duration=int(1000/video_fps), loop=0)
    elif ext in ["mp4", "avi"]:
        imgs = [img for img in result]
        video_clip = ImageSequenceClip(imgs, fps=video_fps)
        video_clip.write_videofile(output_path, video_fps)
    return

def set_seed_logger(args):
    if os.path.exists(args.output_dir)==False and dist.gen_rank()==0:
        os.makedirs(args.output_dir)
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
    
    if dist.get_rank() == 0:
        logger.log("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.log("  <<< {}: {}".format(key, args.__dict__[key]))
    return args

def set_seed_logger_random(args):
    '''
    training or evaluation on multiple GPUs requires different randomness
    '''
    if os.path.exists(args.output_dir)==False and dist.get_rank()==0:
        os.makedirs(args.output_dir)
    # random.seed(args.seed)
    # os.environ['PYTHONHASHSEED'] = str(args.seed)
    # np.random.seed(args.seed)
    # th.manual_seed(args.seed)
    # th.cuda.manual_seed(args.seed)
    # th.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
    
    if dist.get_rank() == 0:
        logger.log("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.log("  <<< {}: {}".format(key, args.__dict__[key]))
    return args


