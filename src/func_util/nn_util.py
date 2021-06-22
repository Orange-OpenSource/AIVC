"""Gathers useful functions to monitor a deep neural network
"""

import numpy as np
import torch
from collections import namedtuple
from torch.autograd import Variable

from func_util.console_display import print_log_msg


def cat_N_yuv_dic(list_yuv_dic):
    """
    Concatenate all YUV dictionnaries contained in list_yuv_dic.
    Return a unique YUV dictionnary, where everything has been
    concatenated along the batch dimension.

    # ! Each Y, U, V channel in the list_yuv_dic must be of dimension 4
    # ! This should propagate the gradient properly
    """

    if len(list_yuv_dic) == 0:
        print_log_msg('ERROR', 'cat_N_yuv_dic', 'list_yuv_dic', 'empty')

    result = {}

    for c in ['y', 'u', 'v']:
        list_yuv_dic_c = [cur_frame.get(c) for cur_frame in list_yuv_dic]
        result[c] = torch.cat(list_yuv_dic_c, dim=0)

    return result


def add_dummy_batch_dim(x):
    """
    Transform x a 3-D tensor [C, H, W] into a 4-D tensor [1, C, H, W]
    """
    return x.view(1, x.size()[0], x.size()[1], x.size()[2])


def add_dummy_batch_dim_dic(x):
    """
    Add a dummy batch index to all entries in a YUV dic
    """
    for k in ['y', 'u', 'v']:
        x[k] = add_dummy_batch_dim(x.get(k))
    return x


def dic_zeros_like(x):
    """
    Generate a dictionnary where all entries are tensors full
    of zeros, with the same size as x.
    Push this dictionnary on the same device than x.
    """

    result = {}
    for k in x:
        result[k] = torch.zeros_like(x.get(k), device=x.get(k).device)

    return result


def set_dic_to_zero(x):
    """
    Set all channels to zeros (but keep the dimension) in a YUV dic
    """
    x_device = x.get('y').device
    for k in ['y', 'u', 'v']:
        x[k] = torch.zeros_like(x.get(k), device=x_device)
    return x


def clamp_dic(x, min_val, max_val):
    for k in ['y', 'u', 'v']:
        x[k] = torch.clamp(x.get(k), min_val, max_val)
    return x


def crop_dic(dic_to_crop, dic_target_size):
    for k in ['y', 'u', 'v']:
        tgt = dic_target_size.get(k)
        tgt_siz_2 = tgt.size()[2]
        tgt_siz_3 = tgt.size()[3]

        dic_to_crop[k] = dic_to_crop.get(k)[:, :, :tgt_siz_2, :tgt_siz_3]
    return dic_to_crop


def push_dic_to_device(x, device):
    for k in ['y', 'u', 'v']:
        if not(x.get(k) is None):
            x[k] = x.get(k).to(device, non_blocking=True)
    return x


def push_gop_to_device(x, device):
    """
    Push a dic (of frames) of dic (of channel) x to the device
    """
    for f in x:
        x[f] = push_dic_to_device(x.get(f), device)

    return x


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def convert_tensor_to_dic(x):
    """Convert a 4-D tensor with 3 channels to a dic
    of 3 4-D tensor with one channel: y, u and v
    """
    B = x.size()[0]
    H = x.size()[2]
    W = x.size()[3]

    res = {
        'y': x[:, 0, :, :].view(B, 1, H, W),
        'u': x[:, 1, :, :].view(B, 1, H, W),
        'v': x[:, 2, :, :].view(B, 1, H, W)
    }

    return res


def get_value(key, dic, default_dic):
    """
    Return value of the entry <key> in <dic>. If it is not defined
    in dic, look into <default_dic>
    """

    v = dic.get(key)

    if v is None:
        if key in default_dic:
            v = default_dic.get(key)
        else:
            print_log_msg(
                'ERROR', 'get_param', 'key not in default_dic', key
            )

    return v


def override_default_dic(dic, default_dic):
    """
    Override each entry of default dic with those of dic if it exists.
    """

    for k in dic:
        default_dic[k] = dic.get(k)

    return dic
