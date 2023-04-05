
import random

from PIL import Image
import cv2
import blobfile as bf

import numpy as np
from torch.utils.data import DataLoader, Dataset
from einops import rearrange, repeat
from torchvision import transforms as T
import torch as th

def load_data(
    *,
    data_dir,
    frame_num,
    batch_size,
    image_size,
    class_cond=False,
    order_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    num_workers=0,
    frame_gap=8
    
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    data_dir_splits = data_dir.split(',')

    all_files = []
    for data_dir_split in data_dir_splits:
        all_files.extend(_list_video_files_recursively(data_dir_split)) 
    
    print(f"len(data loader):{len(all_files)}")
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [path.split("/")[-2] for path in all_files]
        class_labels = set(class_names)
        sorted_classes = {x: i for i, x in enumerate(sorted(class_labels))}
        classes = [sorted_classes[x] for x in class_names]

      
        print(f"len(data loader classes):{len(class_labels)}")
   
    dataset = VideoDataset(
        all_files,
        image_size,
        frame_num,
        classes=classes,
        shard=0,
        num_shards=1,
        random_crop=random_crop,
        random_flip=random_flip,
        frame_gap = frame_gap,
        order_cond = order_cond
    )
    
    # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=True
        )
        
    while True:
        yield from loader


def _list_video_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["avi", "gif", "mp4","png"]:
            
            results.append(full_path)
        elif bf.isdir(full_path):
            
            results.extend(_list_video_files_recursively(full_path))
    return results


class VideoDataset(Dataset):
    def __init__(
        self,
        video_paths,
        resolution,
        frame_num,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=False,
        frame_gap=8,
        order_cond=False
    ):
        super().__init__()
        self.resolution = resolution
        self.local_videos = video_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.frame_num = frame_num
        self.frame_gap = frame_gap
        self.order_cond = order_cond

    def __len__(self):
        return len(self.local_videos)

    def resize_img(self, img):# size:64:64
        '''
        resize img to target_size with padding
        '''
        
        old_size = img.size
        ratio = min(float(self.resolution)/(old_size[i]) for i in range(len(old_size)))
        new_size = tuple([int(i*ratio) for i in old_size])
        
        img = img.resize((new_size[1], new_size[0]),Image.BICUBIC) #cv2.resize(img,(new_size[1], new_size[0]))
        img = np.array(img)
        pad_w = self.resolution - new_size[1]
        pad_h = self.resolution- new_size[0]
        top,bottom = pad_h//2, pad_h-(pad_h//2)
        left,right = pad_w//2, pad_w -(pad_w//2)
        img_new = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,None,(0,0,0))
        return img_new
            
    def _get_gif(self, path):
        with bf.BlobFile(path, "rb") as f:
            pil_images = Image.open(f)
            pil_images = list(map(self.resize_img, seek_all_images(pil_images, channels = 3)))
        return pil_images

    def _get_vid(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        count = 0   
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == False:
                break
            
            img =Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            frames.append(self.resize_img(img))
            count+=1
        return frames

    def __getitem__(self, idx):
        path = self.local_videos[idx]
        # load gif
        post_fix=path.split('.')[-1]
        if post_fix == 'gif':
            video = th.tensor(np.array(self._get_gif(path)))
        elif post_fix in ['avi', 'mp4']:
            video = th.tensor(np.array(self._get_vid(path)))
        elif post_fix in ['png']:
            video = th.tensor(np.array(Image.open(path).convert('RGB'))) # H,W,C
            f_num = video.shape[0] // self.resolution 
            video = rearrange(video, "(f1 h) (f2 w) c -> (f1 f2) h w c", f1=f_num, f2=f_num)
       
        video= video.permute([0,3,1,2])#[B,C,H,W]
 
        if len(video) < self.frame_num:
            append = self.frame_num - len(video) 
            video = th.cat([video, video[-1:].repeat(append,1,1,1)],dim=0)
        elif  len(video) >= self.frame_num and len(video) <=  self.frame_num * self.frame_gap:
            indices = np.linspace(0, len(video)-1, self.frame_num)
            video = th.stack([video[int(indice)] for indice in indices], dim=0)
        else:
            start = random.randint(0, len(video) - self.frame_num * self.frame_gap - 1)
            video = video[start : start + self.frame_num * self.frame_gap : self.frame_gap]
        
       

        video_after_process = video#.float() / 127.5 - 1 #0-1
    
      
        return video_after_process

def seek_all_images(img, channels = 3):

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert("RGB")
        except EOFError:
            break
        i += 1

def load_data_for_worker(base_samples, image_size, frame_num, frame_gap=1, class_cond=False, batch_size=1):
    if base_samples.endswith('npz'):
        with bf.BlobFile(base_samples, "rb") as f:
            obj = np.load(f)
            image_arr = obj["arr_0"]
            if class_cond:
                label_arr = obj["arr_1"]
        
        while True:
            for i in range(len(image_arr)):
                video= image_arr[i]
                #b,f,h,w,c->b,f
                yield video
    else:
        
        dataset = load_data(
            data_dir=base_samples,
            batch_size=batch_size,
            frame_num=frame_num,
            image_size=image_size,
            class_cond=class_cond,
            num_workers=4,
            frame_gap=frame_gap,
            deterministic=True
        )
       

        for batchdata in dataset:
            # import pdb; pdb.set_trace()
            batchdata= batchdata.permute(0, 1, 3, 4, 2)#.astype('uint8')  #[batchsize, frame, W, H, C]
            if batchdata.shape[1] < 16:
                batchdata =th.cat([batchdata, batchdata[:,-1:,].repeat(1,8,1,1,1)], dim=1)

                # video = th.cat([video, video[-1:].repeat(append,1,1,1)],dim=0)
            yield batchdata

              

