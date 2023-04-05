#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys

import numpy as np

import chainer
import chainer.cuda
import cv2 as cv
# import cupy
from c3d.c3d_ft import C3DVersion1
from chainer import Variable
from chainer import cuda
from tqdm import tqdm

sys.path.insert(0, '.')  # isort:skip
from util import load_data_for_worker

def calc_inception(ys):
    N, C = ys.shape
    p_all = np.mean(ys, axis=0, keepdims=True)
    kl = np.sum(ys * np.log(ys + 1e-7) - ys * np.log(p_all + 1e-7)) / N
    return np.exp(kl)


def main():
    parser = argparse.ArgumentParser(description='inception score')
    parser.add_argument("--ref_batch", type=str, default="/data6/rld/data/UCF-101/train/ApplyEyeMakeup", help="path to reference batch npz file")
    parser.add_argument("--size", type=int, default=128, help="path to sample batch npz file")
    parser.add_argument("--frame_num",type=int, default=16, help="path to sample batch npz file")
    parser.add_argument("--sample_frame_gap",type=int, default=1, help="path to sample batch npz file")
    parser.add_argument("--sample_num", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument('--result_dir', type=str, default="./outputs/video-eval/")
    parser.add_argument('--devices', type=str, default='0')
    parser.add_argument('--mean', type=str, default='models/mean2.npz')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--interpolation', type=str, default='INTER_CUBIC')
    args = parser.parse_args()

    np.random.seed(args.seed)

    inter_method = args.interpolation
    args.interpolation = getattr(cv, args.interpolation)

    cuda.get_device(args.devices).use()
    # cupy.random.seed(args.seed)
    xp = chainer.cuda.cupy

    c3dmodel = C3DVersion1()
    c3dmodel.to_gpu()

    # load model

    mean = np.load(args.mean)['mean'].astype('f')
    mean = mean.reshape((3, 1, 16, 128, 171))[:, :, :, :, 21:21 + 128]

    sample_loader = load_data_for_worker(base_samples=args.sample_batch, image_size=args.size, \
        frame_num=args.frame_num, frame_gap= args.sample_frame_gap, batchsize=args.batchsize )
    # generator
    ys = []
    for batch_data in tqdm(sample_loader): # b f h w 3
        n, f, h, w, c = batch_data.shape
        batch_data = batch_data.reshape(n * f, h, w, c)
        x_ = np.zeros((n * f, 128, 128, 3))
        for t in range(n * f):
            x_[t] = np.asarray(
                cv.resize(x[t], (args.size, args.size), interpolation=args.interpolation))# [n*f, 128, 128, 3]

        x = x_.transpose(3, 0, 1, 2).reshape(c, n, f, args.size, args.size)
        x = x[::-1] - mean  # mean file is BGR-order while model outputs RGB-order
        x = x[:, :, :, 8:8 + 112, 8:8 + 112].astype('f')
        x = x.transpose(1, 0, 2, 3, 4)
        with chainer.using_config('train', False) and \
                chainer.no_backprop_mode():
            # C3D takes an image with BGR order
            y = c3dmodel(Variable(xp.asarray(x)),
                         layers=['prob'])['prob'].data.get()
            ys.append(y)
    ys = np.asarray(ys).reshape((-1, 101))

    
    score = calc_inception(ys)
    with open(f'{args.result_dir}/evaluation-{args.sample_batch}.log'.format(args.iter, inter_method), 'a+') as fp:
        print(args.result_dir, args.iter, args.calc_iter, args.mean, score, file=fp)
        print('IS score:{}'.format(score))

    return 0


if __name__ == '__main__':
    sys.exit(main())
