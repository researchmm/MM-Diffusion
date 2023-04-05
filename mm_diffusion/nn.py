"""
Various utilities for neural networks.
"""

import math

import torch as th
import torch.nn as nn
from einops import rearrange, repeat

# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)
        
class GroupNorm32(nn.Module):
    def __init__(self, group, channel ):
        super(GroupNorm32, self).__init__()
        self.channel = channel
        self.GroupNorm = nn.GroupNorm(group, channel)

    def forward(self, x):
        rearrange_flag = False
        if x.shape[1] != self.channel and x.dim()==5:
            b,f,c,h,w =x.shape
            x = rearrange(x, 'b t c h w -> b c t h w')
            rearrange_flag = True

        x = self.GroupNorm(x.float()).type(x.dtype)

        if rearrange_flag:
            x = rearrange(x, 'b c t h w -> b t c h w', b=b)
        return x

class ImgGroupNorm(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class GroupNorm32_3d(nn.Module):
    def __init__(self, group, channel, batch_size):
        super(GroupNorm32_3d, self).__init__()
        self.batch_size = batch_size
        self.GroupNorm = nn.GroupNorm(group, channel)

    def forward(self, x):
    
        input_cluster = True
        if x.shape[0] > self.batch_size:
            if x.dim() == 3:
                # b_x, c_x, w_x = x.shape
                h = rearrange(x, '(b t) c h -> b c h t' , b=self.batch_size)
            elif x.dim() == 4:
                # b_x, c_x, w_x, h_x = x.shape
                h = rearrange(x, '(b t) c h w -> b c h w t' , b=self.batch_size)
            elif x.dim()==5:
                # b_x, c_x, w_x, h_x, o_x = x.shape
                h = rearrange(x, '(b t) c h w o -> b c h w o t' , b=self.batch_size)
            else:
                raise NotImplementedError
        else:
            input_cluster = False
            
            h = rearrange(x, 'b t c h w -> b c h w t' )

        

        h = self.GroupNorm.forward(h.float()).type(x.dtype)
        if input_cluster:
            if h.dim() == 5:
                h = rearrange(h, 'b c h w t -> (b t) c h w')
            elif h.dim() == 4:
                h = rearrange(h, 'b c h t -> (b t) c h')
            elif h.dim() == 6:
                h = rearrange(h, 'b c h w o t -> (b t) c h w o')
            else:
                raise NotImplementedError
        else:
            h =  rearrange(h, 'b c h w t -> b t c h w' )

        return h


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class temporal_conv(nn.Module):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    def __init__(self,*args, **kwargs):
        self.conv = nn.Conv1d(*args, **kwargs)
    def forward(x):
        
        return self.conv(x)
    


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization_3d(channels, batch_size):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    
    return GroupNorm32_3d(32, channels, batch_size)
def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)

def Imgnormalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return ImgGroupNorm(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def temporalstep_embedding(timesteps, dim, max_period=10):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
