# Software Name: AIVC
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

"""Module gathering useful mathematical functions, mainly used for rate estimation 
"""

import torch
import numpy as np
from torch import log10

from func_util.img_processing import get_y_u_v
from func_util.ms_ssim import msssim

LOG_NUM_STAB = 2 ** (-16)
DIV_NUM_STAB = 1e-12

PROBA_MIN = LOG_NUM_STAB
PROBA_MAX = 1.0

# Log var cannot be > 10
# (i.e: sigma < exp(0.5 * 10) = 148,41)
LOG_VAR_MAX = 10.
# Log var cannot be < -18.4207
# (i.e: sigma > exp(0.5 * -18.4207) = 0.0001)
LOG_VAR_MIN = -18.4207


def xavier_init(param_shape):
    """Performs manually the xavier weights initialization namely:

    Weights are sampled from N(0, sqrt(2 / nb_weights))

    Arguments:
    param_shape {[torch.Size]} --
        [A torch.Size object describing the shape of the parameters
        to initialize]

    Returns:
        [torch.tensor] -- [the initialized parameters]
    """
    param_shape = torch.Size(param_shape)
    xav_norm = torch.sqrt(torch.tensor([2.0]) / param_shape.numel())
    init_w = torch.randn(param_shape, requires_grad=True) * xav_norm
    return init_w


def compute_mse_psnr_dic(x_hat, code, max_value=1.):
    x_hat_y, x_hat_u, x_hat_v = get_y_u_v(x_hat)
    code_y, code_u, code_v = get_y_u_v(code)

    squared_err = ((x_hat_y - code_y) ** 2).sum()
    squared_err += ((x_hat_u - code_u) ** 2).sum()
    squared_err += ((x_hat_v - code_v) ** 2).sum()

    nb_values = x_hat_y.numel() + x_hat_u.numel() + x_hat_v.numel()
    mse = squared_err / nb_values
    psnr = 20. * log10(torch.tensor([max_value])) -\
        10. * log10(torch.tensor([mse]))

    return squared_err, mse, psnr


def compute_ms_ssim_dic(x_hat, code, max_value=1.):
    """
        Compute (weighted) MS-SSIM for two YUV dictionnaries.

        Return one value: the ms_ssim
    """
    res = 0
    nb_values = 0
    for x_hat_channel, code_channel in zip(get_y_u_v(x_hat), get_y_u_v(code)):
        nb_values += x_hat_channel.numel()

        channel_ms_ssim = compute_ms_ssim(
            x_hat_channel,
            code_channel,
            max_value=max_value)

        res += channel_ms_ssim * x_hat_channel.numel()

    res /= nb_values
    return res


def compute_ms_ssim(x, y, max_value=1.):
    """
        Compute Multi Scale Structural Similary for two images x, y of shape
        [1, B, H, W].

        It uses the settings allowing to obtain the same results as the
        CLIC leaderboard.
    """

    res = msssim(
        x,
        y,
        window_size=11,
        size_average=True,
        val_range=max_value,
        full=True
    )

    return res


def compute_squared_err(x, y, mode='torch'):
    """
    Return both sum of squared error and mean squarred error between x and y

    x and y can be N-dimensional torch tensor (mode='torch') or 
    np array (mode='array')
    """

    if mode == 'torch':
        num_element = x.numel()
        sq_err = torch.sum((x - y) ** 2)
    elif mode == 'np':
        num_element = x.size
        sq_err = np.sum(np.square(x - y))
    else:
        print('\n[ERROR] compute_squared_err unknown mode!\n')
        return None, None

    mse = sq_err / num_element
    return sq_err, mse


def compute_psnr(mse, max_value=1., mode='torch'):
    """
    Compute PSNR from the Mean Squared Error <mse> and the max_value:

        psnr = 20. * log10(max_value) - 10. * log10(mse)

    log10 operations are realised either with pytorch (mode='torch') or
    numpy (mode='np)
    """

    if mode == 'torch':
        psnr = 20. * log10(torch.tensor([max_value])) -\
            10. * log10(torch.tensor([mse]))
    elif mode == 'np':
        psnr = 20. * np.log10(max_value) - 10. * np.log10(mse)
    else:
        print('\n[ERROR] compute_psnr unknown mode!\n')
        return None
    return psnr
