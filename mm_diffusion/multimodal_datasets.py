from distutils.spawn import spawn
import random
import blobfile as bf
from mpi4py import MPI
import numpy as np
import torch as th
import os
import pickle
import torch as th
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.video_utils import VideoClips
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from moviepy.editor import AudioFileClip

def load_data(
    *,
    data_dir,
    batch_size,
    video_size,
    audio_size,
    deterministic=False,
    random_flip=True,
    num_workers=0,
    video_fps=10,
    audio_fps=None,
    frame_gap=1,
    drop_last=True
):
    """
    For a dataset, create a generator over (audio-video) pairs.

    Each video is an NxFxCxHxW float tensor, each audio is an NxCxL float tensor
   
    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param video_size: the size to which video frames are resized.
    :audio_size:the size to which audio are resized.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
   
    all_files = []
    
    all_files.extend(_list_video_files_recursively(data_dir)) 
    if MPI.COMM_WORLD.Get_rank()==0:
        print(f"len(data loader):{len(all_files)}")
       
    clip_length_in_frames = video_size[0]
    frames_between_clips = 1
    meta_fname = os.path.join(data_dir, f"video_clip_f{clip_length_in_frames}_g{frames_between_clips}_r{video_fps}.pkl")
   
    if not os.path.exists(meta_fname):
        if MPI.COMM_WORLD.Get_rank()==0:
            print(f"prepare {meta_fname}...")
        
        video_clips = VideoClips(
                video_paths=all_files,
                clip_length_in_frames=clip_length_in_frames, #64
                frames_between_clips=frames_between_clips,
                num_workers=16,
                frame_rate = video_fps
            )
        
        if MPI.COMM_WORLD.Get_rank()==0:
            with open(meta_fname, 'wb') as f:
                pickle.dump(video_clips.metadata, f)
            
    else:
        print(f"load {meta_fname}...")
        metadata = pickle.load(open(meta_fname, 'rb'))

        video_clips = VideoClips(video_paths=all_files,
                clip_length_in_frames=clip_length_in_frames, #64
                frames_between_clips=frames_between_clips,
                frame_rate = video_fps,
                _precomputed_metadata=metadata)

    print(f"load {video_clips.num_clips()} video clips from {meta_fname}......")
    dataset = MultimodalDataset(
        video_size = video_size,
        audio_size = audio_size,
        video_clips = video_clips,
        shard = MPI.COMM_WORLD.Get_rank(),
        num_shards = MPI.COMM_WORLD.Get_size(),
        random_flip = random_flip,
        audio_fps = audio_fps,
        frame_gap = frame_gap
    )
    
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=drop_last
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=drop_last
        )
        
    while True:
        yield from loader

def _list_video_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["avi", "gif", "mp4"]:
           
            results.append(full_path)
        elif bf.isdir(full_path):
            
            results.extend(_list_video_files_recursively(full_path))
    return results

class MultimodalDataset(Dataset):
    """
    :param video_size: [F,3,H,W] the size to which video frames are resized.
    :param audio_size: [C,L] the size to which audio are resampled.
    :param video_clips: the meta info package of video clips. 
    :param shard: GPU id, used for allocating videos to different GPUs.
    :param num_shards: GPU num, used for allocating videos to different GPUs.
    :param random_flip: if True, randomly flip the images for augmentation.
    :param audio_fps: the fps of audio.
    """
    def __init__(
        self,
        video_size,
        audio_size,
        video_clips,
        shard=0,
        num_shards=1,
        random_flip=True,
        audio_fps=None,
        frame_gap=1
    ):
        super().__init__()
        self.video_size = video_size#[f,C,H,W]
        self.audio_size = audio_size#[c,len]
        self.random_flip = random_flip
        self.video_clips = video_clips
        self.audio_fps = audio_fps
        self.frame_gap = frame_gap
        self.size = self.video_clips.num_clips()
        self.shuffle_indices = [i for i in list(range(self.size))[shard:][::num_shards]]
        random.shuffle(self.shuffle_indices)

    def __len__(self):
        return len(self.shuffle_indices)
        
    def process_video(self, video):# size:64:64
        '''
        resize img to target_size with padding, 
        augment with RandomHorizontalFlip if self.random_flip is True.

        :param video: ten[f, c, h, w]
        ''' 
        video = video.permute([0,3,1,2])
        old_size = video.shape[2:4]
        ratio = min(float(self.video_size[2])/(old_size[0]), float(self.video_size[3])/(old_size[1]) )
        new_size = tuple([int(i*ratio) for i in old_size])
        pad_w = self.video_size[3] - new_size[1]
        pad_h = self.video_size[2]- new_size[0]
        top,bottom = pad_h//2, pad_h-(pad_h//2)
        left,right = pad_w//2, pad_w -(pad_w//2)
        transform = T.Compose([T.RandomHorizontalFlip(self.random_flip), T.Resize(new_size, interpolation=InterpolationMode.BICUBIC), T.Pad((left, top, right, bottom))])
        video_new = transform(video)
        return video_new
            

    def get_item(self, idx):
   
        while True:
            try:
                video, raw_audio, info, video_idx = self.video_clips.get_clip(idx)
            except Exception:
                idx = (idx + 1) % self.video_clips.num_clips()
                continue
            break
             
        
        if len(video) < self.video_size[0]:
            append = self.video_size[0]- len(video) 
            video = th.cat([video, video[-1:].repeat(append,1,1,1)],dim=0)
        else:
            video = video[:self.video_size[0]]

        video_after_process = self.process_video(video)
        video_after_process = video_after_process.float() / 127.5 - 1 #0-1
 
        video_idx, clip_idx = self.video_clips.get_clip_location(idx)
        duration_per_frame = self.video_clips.video_pts[video_idx][1] - self.video_clips.video_pts[video_idx][0]
        video_fps = self.video_clips.video_fps[video_idx]
        audio_fps = self.audio_fps if self.audio_fps  else info['audio_fps']

        clip_pts = self.video_clips.clips[video_idx][clip_idx]
        clip_pid = clip_pts // duration_per_frame
    
        start_t = (clip_pid[0] / video_fps * 1. ).item()
        end_t = ((clip_pid[-1] + 1) / video_fps * 1. ).item()
       
        video_path = self.video_clips.video_paths[video_idx]
        raw_audio =  AudioFileClip(video_path, fps=audio_fps).subclip(start_t, end_t)
        
        audio = np.zeros(self.audio_size)
        raw_audio= raw_audio.to_soundarray()
        if raw_audio.shape[1] == 2:
            raw_audio = raw_audio[:, 0:1].T # pick one channel
        if  raw_audio.shape[1] < self.audio_size[1]:
            audio[:, :raw_audio.shape[1]] = raw_audio
        elif  raw_audio.shape[1] >= self.audio_size[1]:
            audio = raw_audio[:, :self.audio_size[1]]

        audio = th.tensor(audio)
        
        return video_after_process, audio
    
    def __getitem__(self, idx):
        idx = self.shuffle_indices[idx]
        video_after_process, audio = self.get_item(idx)

        return video_after_process, audio


if __name__=='__main__':
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    from moviepy.audio.AudioClip import AudioArrayClip
    from einops import rearrange
    import torch.nn.functional as F

    audio_fps=16000
    video_fps= 10
    batch_size=4
    seconds = 1.6
    image_resolution=64

    dataset64=load_data(
    data_dir="/data6/rld/data/landscape/test",
    batch_size=batch_size,
    video_size=[int(seconds*video_fps), 3, 64, 64],
    audio_size=[1, int(seconds*audio_fps)],
    frame_gap=1,
    random_flip=False,
    num_workers=0,
    deterministic=True,
    video_fps=video_fps,
    audio_fps=audio_fps
    )

  
    group = 0

    while True:    
        group += 1
        batch_video, batch_audio,  cond= next(dataset64)
   
