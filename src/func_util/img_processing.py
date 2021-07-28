# Software Name: AIVC
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

"""
Several functions related to image processing:
    - conversion between colorspaces
    - scaling
    - incomplete ctu padding
    - normalization
    - slicing a whole picture into ctus
"""
# Built-in modules
import string

# Third party modules
import torch

from PIL import Image       # Used to load picture and transform it to YCbCr
import numpy as np
from torchvision.transforms.functional import to_tensor, to_pil_image
from torch.nn.functional import interpolate
from func_util.nn_util import add_dummy_batch_dim, convert_tensor_to_dic, get_value,\
                              add_dummy_batch_dim_dic

def cast_before_png_saving(param):
    """
    Saving a (float32) tensor to a PNG image with only 256 levels possible
    introduces some distortion. Custom functions save_tensor_as_img or
    save_yuv_separately relies on the PyTorch function to_pil_image.
    This function cast a tensor x as follows:
        x_png = int(255 * x)
    Which is not the best way of doing it as the int() operation performs a
    truncation and not a round, leading to a significant increase of the distortion.

    This function purpose is two-fold:
        - Perform a better rounding operation
        - Return a float32 tensor which is still a floating point tensor in [0, 1],
          but it has only 256 levels. That means that calling the to_pil_image
          function on it will be **perfectly lossless**.

    # ! Before doing the casting, we do a clamp of the input data to keep them in [0, 1]

    # ! Remark: to replicate the old behaviour, replace .round() by .int()
    """

    DEFAULT_PARAM = {
        # Tensor to be casted
        'x': None,
        # What's the type of x. Available:
        #   <tensor> : x is a pytorch tensor of dimension N
        #   <yuv_dic>: x is a dictionnary with 3 entries (Y, U & V)
        #              each of them being a pytorch tensor of dimension N
        'data_type': 'yuv_dic',
    }

    # =========================== RETRIEVE INPUTS =========================== #
    x = get_value('x', param, DEFAULT_PARAM)
    data_type = get_value('data_type', param, DEFAULT_PARAM)
    # =========================== RETRIEVE INPUTS =========================== #

    if data_type == 'tensor':
        x_png = ((255. * torch.clamp(x, 0., 1.) ).round() / 255.).float()

    elif data_type == 'yuv_dic':
        x_png = {}
        for c in ['y', 'u', 'v']:
            x_png[c] = ((255. * torch.clamp(x.get(c), 0., 1.) ).round() / 255.).float()

    return x_png


def load_frames(param):
    """
    Return a dictionnary of dictionnaries structured as:
    result = {
        'frame_0': {'y': tensor, 'u': tensor, 'v': tensor},
        'frame_1': {'y': tensor, 'u': tensor, 'v': tensor},
        'frame_N': {'y': tensor, 'u': tensor, 'v': tensor},
    }

    for N + 1 frames.

    The frame are loaded from the folder whose absolute path is <sequence_path>.
    Inside <sequence_path>, we find 3N PNGs for a N-frame video: 
    <idx_frame>_y.png, <idx_frame>_u.png, <idx_frame>_v.png

    We start loading the frame at <idx_starting_frame> and we load <nb_frame_to_load>.
    """

    DEFAULT_PARAM = {
        # Absolute path of the folder gathering all frames of the sequence to code.
        'sequence_path': None,
        # Index of the first frame to load
        'idx_starting_frame': 0,
        # Number of successive frames to load
        'nb_frame_to_load': 3,
        # When loading <nb_frame_to_load> frames, replace the <nb_pad_frame> frames
        # with the last loaded one. For instance, if <nb_frame_to_load> = 3 and
        # <nb_pad_frame> = 1, we load frame_0 and frame_1, but we don't load frame_2.
        # Instead, we replace it with frame_1. This option is here to deal with 
        # incomplete GOP
        'nb_pad_frame': 0,
        # If True, look for a single png with all 3 color channels instead of 3
        # PNGs for Y, U and V channels.
        'rgb': False,
        # How the (raw) video frames are organized. Available:
        #   <old> : used for the ICLR paper i.e. <sequence_path>/idx_<y,u,v>.png
        #   <clic>: <sequence_path>/<sequence_name>_<padded_idx>_<y,u,v>.png
        'loading_mode': 'old',
    }

    # =========================== RETRIEVE INPUTS =========================== #
    sequence_path = get_value('sequence_path', param, DEFAULT_PARAM)
    idx_starting_frame = get_value('idx_starting_frame', param, DEFAULT_PARAM)
    nb_frame_to_load = get_value('nb_frame_to_load', param, DEFAULT_PARAM)
    nb_pad_frame = get_value('nb_pad_frame', param, DEFAULT_PARAM)
    rgb = get_value('rgb', param, DEFAULT_PARAM)
    loading_mode = get_value('loading_mode', param, DEFAULT_PARAM)
    # =========================== RETRIEVE INPUTS =========================== #

    if not(sequence_path.endswith('/')):
        sequence_path += '/'

    # Loaded frames will be gathered here
    frames = {}

    GOP_idx = 0

    # Loop to load all required frames.
    # GOP_idx is the relative number of the frame inside a GOP (0, 1, 2 ...)
    # absolute_idx is the absolute number of the frame inside the sequence(N, N + 1, N + 2, ...)
    for absolute_idx in range(idx_starting_frame, idx_starting_frame + nb_frame_to_load - nb_pad_frame):
        # Internal name of the frame (i.e. in the code)
        name_frame = 'frame_' + str(GOP_idx)
        # Storage name of the frame (i.e. on the disk)
        if loading_mode == 'old':
            storage_name = sequence_path + str(absolute_idx)
        elif loading_mode == 'clic':
            storage_name = sequence_path + sequence_path.split('/')[-2] + '_' + str(absolute_idx).zfill(5)

        if rgb:
            # No need to add dummy batch dimension, it is already done inside the
            # load RGB as YUV function
            frames[name_frame] = load_RGB_as_YUV420_dic(storage_name)
        else:
            frames[name_frame] = add_dummy_batch_dim_dic(load_YUV_as_dic_tensor(storage_name))

        GOP_idx += 1

    # This is the very last frame of the video. To go from this one to the last needed frames
    # just recopy the last frame
    # ! -1 because of range above which excludes the end. (+ we start the counting of frame number at 0)
    idx_last_loaded_frame = idx_starting_frame + nb_frame_to_load - nb_pad_frame - 1
    # Perform padding by copying the last really loaded frames on all the padded frames
    for i in range(nb_pad_frame):
        name_frame = 'frame_' + str(GOP_idx)
        
        # Storage name of the frame (i.e. on the disk)
        if loading_mode == 'old':
            storage_name = sequence_path + str(idx_last_loaded_frame)
        elif loading_mode == 'clic':
            storage_name = sequence_path + sequence_path.split('/')[-2] + '_' + str(idx_last_loaded_frame).zfill(5)

        if rgb:
            frames[name_frame] = load_RGB_as_YUV420_dic(storage_name)
        else:
            frames[name_frame] = add_dummy_batch_dim_dic(load_YUV_as_dic_tensor(storage_name))
        
        GOP_idx += 1

    return frames


def load_RGB_as_YUV420_dic(path_img):
    """
        From an RGB png file:
        Construct a dic with 3 entries ('y','u', 'v'), each of them
        is a tensor and is loaded from path_img + '.png'.

        ! Return a dictionnary of 4D tensor (i.e. with a dummy batch index)
    """
    img = Image.open(path_img + '.png').convert('YCbCr')

    x = to_tensor(img)
    x = add_dummy_batch_dim(x)
    x = convert_tensor_to_dic(x)

    x['u'] = interpolate_nearest(x.get('u'), scale=0.5)
    x['v'] = interpolate_nearest(x.get('v'), scale=0.5)

    return x


def load_YUV_as_dic_tensor(path_img):
    """
        Construct a dic with 3 entries ('y','u', 'v'), each of them
        is a tensor and is loaded from path_img + key + '.png'.

        ! Return a dictionnary of 3D tensor (i.e. without a dummy batch index)
    """
    dic_res = {}
    key = ['y', 'u', 'v']
    for k in key:
        img = Image.open(path_img + '_' + k + '.png')

        # check if image mode is correct: it should be a one
        # canal uint8 image (i.e. mode L)
        if img.mode != 'L':
            img = img.convert('L')

        dic_res[k] = to_tensor(img)

    return dic_res


def save_tensor_as_img(x, path_img, mode='yuv420'):
    """
    mode='yuv420', 'yuv444' or 'rgb'
    if yuv420:
        x is a dic
    if yuv444:
        x is a dic
    if rgb or yuv444_nodic:
        x is a 4-D or 3-D tensor
    if L:
        x is a 4-D or 3-D tensor which only one color channel
    """
    if mode == 'yuv420' or mode == 'yuv444':
        y, u, v = get_y_u_v(x)

        if y.ndim == 3:
            y = add_dummy_batch_dim(y)
        if u.ndim == 3:
            u = add_dummy_batch_dim(u)
        if v.ndim == 3:
            v = add_dummy_batch_dim(v)

        # ! ----- Every tensor in 4 D ----- ! #
        if mode == 'yuv420':
            u_ups = interpolate_nearest(u, scale=2)
            v_ups = interpolate_nearest(v, scale=2)
        else:
            u_ups = u
            v_ups = v

        # If u and (half resolution has an odd size), upscaling
        # can results in a different dimensions so:
        h = y.size()[2]
        w = y.size()[3]

        # Cases where upscaled version are bigger Y
        # ? Remark u_ups and v_ups have the same size
        u_ups = u_ups[:, :, :h, :w]
        v_ups = v_ups[:, :, :h, :w]
        # ! ----- Every tensor in 4 D ----- ! #

        # ! ----- Every tensor in 3 D ----- ! #
        y = y[0, :, :, :]
        u_ups = u_ups[0, :, :, :]
        v_ups = v_ups[0, :, :, :]

        x = torch.cat((y, u_ups, v_ups), dim=0)
        to_pil_image(x.cpu(), mode='YCbCr').convert('RGB').save(path_img)
        # ! ----- Every tensor in 3 D ----- ! #

    elif mode == 'rgb':
        # Remove batch index if needed
        if x.ndim == 4:
            x = x[0, :, :, :]
        to_pil_image(x.cpu(), mode='RGB').save(path_img)

    elif mode == 'yuv444_nodic':
        # Remove batch index if needed
        if x.ndim == 4:
            x = x[0, :, :, :]
        to_pil_image(x.cpu(), mode='YCbCr').convert('RGB').save(path_img)

    elif mode == 'L':
        # Remove batch index if needed
        if x.ndim == 4:
            x = x[0, :, :, :]
        to_pil_image(x.cpu(), mode='L').save(path_img)


def save_yuv_separately(x, path_img):
    """
    x is a dic with Y, U and V channels.
    Each of them will be saved as <path_img> + '_y.png'
    """
    for key in x:
        cur_channel = x[key]
        if cur_channel.ndim == 4:
            cur_channel = cur_channel[0, :, :, :]

        path_channel = path_img + '_' + key + '.png'
        to_pil_image(cur_channel.cpu(), mode='L').save(path_channel)


def get_y_u_v(x):
    return x.get('y'), x.get('u'), x.get('v')


def interpolate_nearest(x, scale=2):
    """
    Interpolate x to a given upscale, using nearest mode
    """

    return interpolate(x, scale_factor=scale, mode='nearest')
