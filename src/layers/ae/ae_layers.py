# Software Name: AIVC
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

import torch

from torch.nn import ModuleList, Module
from layers.misc.custom_conv_layers import UpscalingLayer, CustomConvLayer
from func_util.img_processing import get_y_u_v, interpolate_nearest
from torch.nn.functional import interpolate

class InputLayer(Module):
    """
    The role of the input layer is to take Y, U, V with
    U and V half the resolution of Y and to output nb_ft channels
    with the same resolution by downscaling Y
    """

    def __init__(self):
        super(InputLayer, self).__init__()

    def forward(self, x):
        y, u, v = get_y_u_v(x)

        u_v_upscaled = interpolate_nearest(
            torch.cat((u, v), dim=1),
            scale=2
        )[:, :, :y.size()[2], : y.size()[3]]

        return torch.cat((y, u_v_upscaled), dim=1)


class OutputLayer(Module):
    def __init__(self, k_size=5):
        super(OutputLayer, self).__init__()

    def forward(self, x):
        res = {}

        res['y'] = x[:, 0, :, :].unsqueeze(1)
        u_v_downscaled = interpolate(
            x[:, 1:, :, :],
            scale_factor=0.5,
            mode='bilinear',
            align_corners=False,
            recompute_scale_factor=False,
        )

        res['u'] = u_v_downscaled[:, 0, :, :].unsqueeze(1)
        res['v'] = u_v_downscaled[:, 1, :, :].unsqueeze(1)
        return res
