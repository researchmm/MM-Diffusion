# MM-Diffusion
This is the official PyTorch implementation of the paper [MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation]().
 
  
## Introduction
We propose the first joint audio-video generation framework named MM-Diffusion that brings engaging watching and listening experiences simultaneously, towards high-quality realistic videos.  MM-Diffusion consists of a sequential multi-modal U-Net. Two subnets for audio and video learn to gradually generate aligned audio-video pairs from Gaussian noises.
<img src="./fig/teaser.png" width=70%>


### Overview
<img src="./fig/MM-UNet2.png" width=100%>


### Visualize
The generated video examples on landscape:

<video src="./fig/landscape.mp4" controls="controls" >您的浏览器不支持播放该视频！</video>

The generated video examples on AIST++:

<video src="./fig/aist++.mp4" controls="controls" >您的浏览器不支持播放该视频！</video>

The generated video examples on Audioset:

<video src="./fig/audioset.mp4" controls="controls" >您的浏览器不支持播放该视频！</video>

## Citation
If you find the code and pre-trained models useful for your research, please consider citing our paper. :blush:
```
@article{ruan2022mmdiffusion,
author = {Ruan, Ludan and Ma, Yiyang and Yang, Huan and He, Huiguo and Liu, Bei and Fu, Jianlong and Yuan, Nicholas Jing and Jin, Qin and Guo, Baining},
title = {MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation},
journal={arXiv preprint},
year = {2022},
month = {December}
}
```