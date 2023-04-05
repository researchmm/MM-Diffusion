'''
evaluate generated samples with AudioCLIP 
'''
import sys; sys.path.extend(['evaluation/AudioCLIP'])
from tqdm import tqdm
import argparse
import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel
import torch as th
import torch.distributed as dist
from evaluations.fvd.fvd import get_fvd_logits, frechet_distance
from evaluations.fvd.download import load_i3d_pretrained
from evaluations.AudioCLIP.get_embedding import load_audioclip_pretrained, get_audioclip_embeddings_scores
from .multimodal_datasets import load_data as load_multimodal_data
from . import dist_util, logger
VIDEO_SIZE=[16,3,224,224]
AUDIO_RATE=44100
AUDIO_SIZE=[1, int(AUDIO_RATE*1.6)]
BATCH_SIZE = 8

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

def load_multimodal_for_worker(base_dir, video_size):
    data = load_multimodal_data(
        data_dir=base_dir,
        batch_size=BATCH_SIZE,
        video_size=video_size,
        audio_size=AUDIO_SIZE,
        num_workers=8,
        frame_gap=1,
        random_flip=False,
        deterministic=True,
        drop_last=False,
        audio_fps=AUDIO_RATE
    )
   
    for video_batch, audio_batch in data:
        video_batch = ((video_batch + 1) * 127.5).clamp(0, 255).to(th.uint8)
        gt_batch = {"video": video_batch, "audio":audio_batch}
        
        yield gt_batch


def eval_multimodal(real_path, fake_path, video_size=[16,3,64,64], eval_num=2048):
    metric = {}
    #################### Load I3D ########################################
    i3d = load_i3d_pretrained(dist_util.dev())
    
    audioclip = load_audioclip_pretrained(dist_util.dev())

    real_loader = load_multimodal_for_worker(real_path, video_size)
    fake_loader = load_multimodal_for_worker(fake_path, video_size)

    fake_video_embeddings = []
    fake_audioclip_video_embeddings = []
    fake_audioclip_audio_embeddings = []
    fake_av_clip_scores = []
   
    for _, sample in enumerate(tqdm(fake_loader)):
        if len(fake_video_embeddings) >= eval_num:break
        # b t h w c
        video_sample = sample["video"].to(dist_util.dev())
        audio_sample = sample["audio"].to(dist_util.dev())

        fake_video_embedding = get_fvd_logits(video_sample, i3d=i3d, device=dist_util.dev())
        gathered_fake_video_embedding = [th.zeros_like(fake_video_embedding) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_fake_video_embedding, fake_video_embedding)  # gather not supported with NCC
        fake_video_embedding = th.cat(gathered_fake_video_embedding, dim=0)   
        fake_video_embeddings = fake_video_embedding if len(fake_video_embeddings) == 0 else th.cat([fake_video_embeddings, fake_video_embedding], dim=0)
        
        fake_audioclip_video_embedding, fake_audioclip_audio_embedding, fake_av_clip_score = get_audioclip_embeddings_scores(audioclip, video_sample, audio_sample)
        
        gathered_fake_audioclip_video_embedding = [th.zeros_like(fake_audioclip_video_embedding) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_fake_audioclip_video_embedding, fake_audioclip_video_embedding)  # gather not supported with NCC
        fake_audioclip_video_embedding = th.cat(gathered_fake_audioclip_video_embedding, dim=0)   
        fake_audioclip_video_embeddings = fake_audioclip_video_embedding if len(fake_audioclip_video_embeddings) == 0 else th.cat([fake_audioclip_video_embeddings, fake_audioclip_video_embedding], dim=0)
        
        gathered_fake_audioclip_audio_embedding = [th.zeros_like(fake_audioclip_audio_embedding) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_fake_audioclip_audio_embedding, fake_audioclip_audio_embedding)  # gather not supported with NCC
        fake_audioclip_audio_embedding = th.cat(gathered_fake_audioclip_audio_embedding, dim=0)   
        fake_audioclip_audio_embeddings = fake_audioclip_audio_embedding if len(fake_audioclip_audio_embeddings) == 0 else th.cat([fake_audioclip_audio_embeddings, fake_audioclip_audio_embedding], dim=0)

        
        gathered_fake_av_clip_score = [th.zeros_like(fake_av_clip_score) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_fake_av_clip_score, fake_av_clip_score)  # gather not supported with NCC
        fake_av_clip_score = th.cat(gathered_fake_av_clip_score, dim=0)   
        fake_av_clip_scores = fake_av_clip_score if len(fake_av_clip_scores) == 0 else th.cat([fake_av_clip_scores, fake_av_clip_score], dim=0)   

        dist.barrier()
        
    real_video_embeddings = []
    real_audioclip_video_embeddings = []
    real_audioclip_audio_embeddings = []
    real_av_clip_scores = []
    for _, sample in enumerate(tqdm(real_loader)):
        if len(real_video_embeddings) >= eval_num:break
        # b t h w c
        video_sample = sample["video"].to(dist_util.dev())
        audio_sample = sample["audio"].to(dist_util.dev())

        real_video_embedding = get_fvd_logits(video_sample, i3d=i3d, device=dist_util.dev())
        gathered_real_video_embedding = [th.zeros_like(real_video_embedding) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_real_video_embedding, real_video_embedding)  # gather not supported with NCC
        real_video_embedding = th.cat(gathered_real_video_embedding, dim=0)   
        real_video_embeddings = real_video_embedding if len(real_video_embeddings) == 0 else th.cat([real_video_embeddings, real_video_embedding], dim=0)
        
        real_audioclip_video_embedding, real_audioclip_audio_embedding, real_av_clip_score = get_audioclip_embeddings_scores(audioclip, video_sample, audio_sample)
        
        gathered_real_audioclip_video_embedding = [th.zeros_like(real_audioclip_video_embedding) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_real_audioclip_video_embedding, real_audioclip_video_embedding)  # gather not supported with NCC
        real_audioclip_video_embedding = th.cat(gathered_real_audioclip_video_embedding, dim=0)   
        real_audioclip_video_embeddings = real_audioclip_video_embedding if len(real_audioclip_video_embeddings) == 0 else th.cat([real_audioclip_video_embeddings, real_audioclip_video_embedding], dim=0)
        
        gathered_real_audioclip_audio_embedding = [th.zeros_like(real_audioclip_audio_embedding) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_real_audioclip_audio_embedding, real_audioclip_audio_embedding)  # gather not supported with NCC
        real_audioclip_audio_embedding = th.cat(gathered_real_audioclip_audio_embedding, dim=0)   
        real_audioclip_audio_embeddings = real_audioclip_audio_embedding if len(real_audioclip_audio_embeddings) == 0 else th.cat([real_audioclip_audio_embeddings, real_audioclip_audio_embedding], dim=0)

        
        gathered_real_av_clip_score = [th.zeros_like(real_av_clip_score) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_real_av_clip_score, real_av_clip_score)  # gather not supported with NCC
        real_av_clip_score = th.cat(gathered_real_av_clip_score, dim=0)   
        real_av_clip_scores = real_av_clip_score if len(real_av_clip_scores) == 0 else th.cat([real_av_clip_scores, real_av_clip_score], dim=0)   
        dist.barrier()
   
    
    fake_video_embeddings = fake_video_embeddings[:eval_num]
    fake_audioclip_video_embeddings = fake_audioclip_video_embeddings[:eval_num]
    fake_audioclip_audio_embeddings = fake_audioclip_audio_embeddings[:eval_num]
    fake_av_clip_scores = fake_av_clip_scores[:eval_num]

    real_video_embeddings = real_video_embeddings[:eval_num]
    real_audioclip_video_embeddings = real_audioclip_video_embeddings[:eval_num]
    real_audioclip_audio_embeddings = real_audioclip_audio_embeddings[:eval_num]
    real_av_clip_scores = real_av_clip_scores[:eval_num]


    if dist.get_rank()==0:
        logger.log(f"evaluate for {len(real_video_embeddings)} samples")

    fvd = frechet_distance(fake_video_embeddings.clone().detach(), real_video_embeddings.clone().detach())
    kvd = polynomial_mmd(fake_video_embeddings.clone().detach().cpu().numpy(), real_video_embeddings.detach().cpu().numpy())
    #clip_fvd = frechet_distance(fake_audioclip_video_embeddings.clone().detach(), real_audioclip_video_embeddings.clone().detach())
    #clip_kvd = polynomial_mmd(fake_audioclip_video_embeddings.clone().detach().cpu().numpy(), real_audioclip_video_embeddings.detach().cpu().numpy())
    
    clip_fad = frechet_distance(fake_audioclip_audio_embeddings.clone().detach(), real_audioclip_audio_embeddings.clone().detach())
    #clip_kad = polynomial_mmd(fake_audioclip_audio_embeddings.clone().detach().cpu().numpy(), real_audioclip_audio_embeddings.detach().cpu().numpy())
    #clip_av_score = fake_av_clip_scores.mean()
    metric["fvd"] = fvd.item()
    metric["kvd"] = kvd.item()
    #metric["clip_fvd"] = clip_fvd.item()
    #metric["clip_kvd"] = clip_kvd.item()

    metric["fad"] = clip_fad.item() * 1000
    #metric["clip_kad"] = clip_kad.item()
    #metric["clip_av_score"] =clip_av_score.item()

    return metric
   
def main(
    ):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_dir", type=str, default="/data6/rld/data/landscape/train", help="path to reference batch npz file")
    parser.add_argument("--fake_dir", type=str, default="/data6/rld/data/landscape/traine", help="path to sample batch npz file")
    parser.add_argument("--output_dir", type=str, default="../outputs/video-eval/debug", help="" )
    parser.add_argument("--sample_num", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--devices", type=str, default="G8")
    args = parser.parse_args()

    dist_util.setup_dist(args.devices)
    logger.configure(dir=args.output_dir, log_suffix="_device")
    metric= eval_multimodal(args.ref_dir, args.fake_dir, eval_num=args.sample_num)
    print(metric)

        
if __name__ == '__main__':
    main()

