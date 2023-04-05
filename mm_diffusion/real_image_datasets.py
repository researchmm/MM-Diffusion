from configparser import Interpolation
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
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=False,
    num_workers=0,
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
        all_files.extend(_list_image_files_recursively(data_dir_split)) 

    if MPI.COMM_WORLD.Get_rank()==0:
        print(f"len(data loader):{len(all_files)}")
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        
        class_names = [path.split("/")[-2] for path in all_files]
        class_labels = set(class_names)
        sorted_classes = {x: i for i, x in enumerate(sorted(class_labels))}
        classes = [sorted_classes[x] for x in class_names]

        if MPI.COMM_WORLD.Get_rank()==0:
            print(f"len(data loader classes):{len(class_labels)}")
    dataset = RealImageDataset(
        image_size,
        all_files,
        classes=classes,
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


class RealImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=False,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
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

    def noise_img(self, img):
        '''
        add gaussian noise to img
        '''
        if random.random() < 0.5:
            img = img.astype(np.float32)
            h, w, c = img.shape
            sigma = random.uniform(0, 20)
            noise = np.random.randn(h, w, c) * sigma
            img_new = img + noise
            # img_new = np.clip(img_new.round(), 0, 255)
            # img_new = img_new.astype(np.uint8)
        else:
            return img
        return img_new

    def jpeg_img(self, img):
        '''
        add jpeg artifact to img
        '''
        if random.random() < 0.5:
            quality = int(random.uniform(20, 80))
            data = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, quality))[1]
            img_new = cv2.imdecode(data, cv2.IMREAD_COLOR)
        else:
            return img
        return img_new

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        hr = self.resize_img(np.array(pil_image))
        lr = cv2.resize(hr, (64, 64), interpolation=cv2.INTER_CUBIC)
        lr = self.noise_img(lr)
        lr = self.jpeg_img(lr)
        sr = cv2.resize(lr, (256, 256), interpolation=cv2.INTER_CUBIC)

        if self.random_flip and random.random() < 0.5:
            hr = hr[:, ::-1]
            lr = lr[:, ::-1]
            sr = sr[:, ::-1]

        hr = hr.astype(np.float32) / 127.5 - 1
        lr = lr.astype(np.float32) / 127.5 - 1
        sr = sr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(lr, [2, 0, 1]), np.transpose(hr, [2, 0, 1]), np.transpose(sr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
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