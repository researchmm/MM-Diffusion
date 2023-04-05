import sys,os
sys.path.append(os.path.dirname (os.path.dirname (os.path.abspath (__file__))))
import argparse
from mm_diffusion import dist_util, logger
from mm_diffusion.evaluator import eval_multimodal
from mm_diffusion.common import  delete_pkl


# command: mpiexec -n 4 python py_scripts/eval.py --devices 0,1,2,3

def main(
    ):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_dir", type=str, default="/home/v-zixituo/rld/data/landscape/train", help="path to reference batch npz file")
    parser.add_argument("--fake_dir", type=str, default="/home/v-zixituo/rld/outputs/MM-Diffusion/samples/video-sample-sr/landscape_16x64x64_bs128_res2_channel128_linear1000_att248_dropout0.1/ema_0.9999_100000.pt/original", help="path to sample batch npz file")
    parser.add_argument("--output_dir", type=str, default="/home/v-zixituo/rld/outputs/MM-Diffusion/video-eval/debug", help="" )
    parser.add_argument("--sample_num", type=int, default=2048)
    parser.add_argument("--devices", type=str, default="G8")
    args = parser.parse_args()

    dist_util.setup_dist(args.devices)
    logger.configure(dir=args.output_dir, log_suffix="_val")

    metric = eval_multimodal(args.ref_dir, args.fake_dir, eval_num=args.sample_num)
    logger.log(f"metric:{metric}")
    delete_pkl(args.fake_dir)    
        
if __name__ == '__main__':
    main()

