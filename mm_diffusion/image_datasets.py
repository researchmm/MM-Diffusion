import math
import random
from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch as th
import cv2

def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    num_workers=0,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    :param num_workers: the number of workers to use for loading data.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    data_dir_splits = data_dir.split(',')

    all_files = []
    for data_dir_split in data_dir_splits:
        all_files.extend(_list_image_files_recursively(data_dir_split)) 

    if MPI.COMM_WORLD.Get_rank()==0:
        print(f"len(data loader):{len(all_files)}")
   
    dataset = ImageDataset(
        image_size,
        all_files,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
        )
        
    while True:
        yield from loader


def _list_image_files_recursively(data_dir, frame_gap=1):
    results = []
    
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png"]:
            results.append(full_path)
            
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path, frame_gap))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=False,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)
        
    def resize_img(self, img):# size:64:64
        '''
        resize img to target_size with padding
        '''
        old_size = img.shape[:2]
        ratio = min(float(self.resolution)/(old_size[i]) for i in range(len(old_size)))
        new_size = tuple([int(i*ratio) for i in old_size])
        img = cv2.resize(img, (new_size[1], new_size[0]), interpolation=cv2.INTER_CUBIC)
        pad_w = self.resolution - new_size[1]
        pad_h = self.resolution- new_size[0]
        top,bottom = pad_h//2, pad_h-(pad_h//2)
        left,right = pad_w//2, pad_w -(pad_w//2)
        img_new = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,None,(0,0,0))
        return img_new

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        arr = self.resize_img(np.array(pil_image))
        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]
        arr = arr.astype(np.float32) / 127.5 - 1

        
        return np.transpose(arr, [2, 0, 1])

if __name__=='__main__':
   
    dataset=load_data(
    data_dir="../../data/ucf101_jpg/v_ApplyEyeMakeup_g01_c01",
    batch_size=8,
    image_size=256,
    frame_gap=8,
    random_flip=True)
    while True:
        import pdb; pdb.set_trace()
        batch, cond = next(dataset)
        batch = ((batch + 1) * 127.5).clamp(0, 255).to(th.uint8)
        images = batch.reshape(-1,3, 256, 256)
        
        images = images.permute(0,2,3,1)
        for ind, image in enumerate(images):
            out_path = f"{ind}.jpg"
            Image.fromarray(image.numpy()).convert('RGB').save(out_path)
          
