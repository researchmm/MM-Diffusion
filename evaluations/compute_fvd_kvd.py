from pkg_resources import yield_lines
import click
import sys; sys.path.extend(['.', 'src'])
from tqdm import tqdm

import argparse
import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel
import torch
from util import load_data_for_worker


from fvd.fvd import get_fvd_logits, frechet_distance
from fvd.download import load_i3d_pretrained



def polynomial_mmd(X, Y):
    m = X.shape[0]
    n = Y.shape[0]

    # compute kernels
    K_XX = polynomial_kernel(X)
    K_YY = polynomial_kernel(Y)
    K_XY = polynomial_kernel(X, Y)

    # compute mmd distance
    K_XX_sum = (K_XX.sum() - np.diagonal(K_XX).sum()) / (m * (m - 1))
    K_YY_sum = (K_YY.sum() - np.diagonal(K_YY).sum()) / (n * (n - 1))
    K_XY_sum = K_XY.sum() / (m * n)

    mmd = K_XX_sum + K_YY_sum - 2 * K_XY_sum

    return mmd



def main(
    ):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_batch", type=str, default="/data6/rld/data/UCF-101/train/ApplyEyeMakeup", help="path to reference batch npz file")
    parser.add_argument("--sample_batch", type=str, default="/data6/rld/data/UCF-101/train/ApplyEyeMakeup", help="path to sample batch npz file")
    parser.add_argument("--size", type=int, default=128, help="path to sample batch npz file")
    parser.add_argument("--frame_num",type=int, default=16, help="path to sample batch npz file")
    parser.add_argument("--sample_frame_gap",type=int, default=8, help="path to sample batch npz file")
    parser.add_argument("--sample_num", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    device = torch.device('cuda')
    #################### Load I3D ########################################
    i3d = load_i3d_pretrained(device)
    #################### Compute FVD ###############################
    
    fvds = []
    kvds = []  
    fvd, kvd = eval_fvd(i3d, args, device)
    fvds.append(fvd)
    kvds.append(kvd)

    fvd_mean = np.mean(fvds)
    kvd_mean = np.mean(kvds)
    fvd_std = np.std(fvds)
    kvd_std = np.std(kvds)

    print(f"Final FVD {fvd_mean:.2f} +/- {fvd_std:.2f}")
    print(f"Final KVD {kvd_mean:.2f} +/- {kvd_std:.2f}")

def eval_fvd(i3d, args, device):
    sample_loader = load_data_for_worker(base_samples=args.sample_batch, image_size=args.size, \
        batch_size=args.batch_size, frame_num=args.frame_num)
    ref_loader = load_data_for_worker(base_samples=args.ref_batch, image_size=args.size, \
        batch_size = args.batch_size, frame_num=args.frame_num, frame_gap = args.sample_frame_gap)
        
    
    print("get real embeddings...")
    real_embeddings = []
    for id,ref in enumerate(ref_loader):
        if id >=args.sample_num:break
        real = ref # BCTHW -> BTHWC
        import pdb; pdb.set_trace()
        real_embeddings.append(get_fvd_logits(real, i3d=i3d, device=device))
    real_embeddings = torch.cat(real_embeddings)

    print("get fake embeddings...")
    fake_embeddings= []
    for id, sample in enumerate(tqdm(sample_loader)):
        if id >=args.sample_num:break
        
        # b t h w c
        fake_embeddings.append(get_fvd_logits(sample, i3d=i3d, device=device))
    fake_embeddings = torch.cat(fake_embeddings)

    

    fvd = frechet_distance(fake_embeddings.clone().detach(), real_embeddings.clone().detach())
    kvd = polynomial_mmd(fake_embeddings.clone().detach().cpu().numpy(), real_embeddings.detach().cpu().numpy())
    return fvd.item(), kvd.item()


if __name__ == '__main__':
    main()
