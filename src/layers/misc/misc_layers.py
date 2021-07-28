# Software Name: AIVC
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

import torch
import torch.utils.data
from torch import nn
from torch.autograd import Function
from torch.nn import Module, Conv2d, ReplicationPad2d,\
                     LeakyReLU, Sequential, ReLU, ConvTranspose2d
import numpy as np
import torch.nn.functional as F
from func_util.math_func import LOG_VAR_MIN, LOG_VAR_MAX

"""
This module implements the Generalized Divise Normalization (GDN) Transform,
proposed by Ballé et al. in:
    http://www.cns.nyu.edu/pub/lcv/balle16a-reprint.pdf, 2016.

This non-linear reparametrization of a linearly transformed vector y = f(x)
(where f is a convolutional or fully connected layer) acts as a non linearity,
replacing the ReLU.
"""

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class LowerBound(Function):

    def __init__(self):
        super(LowerBound, self).__init__()

    @staticmethod
    def forward(ctx, inputs, bound):
        # ! Memory transfer, use device=inputs.device
        b = torch.ones(inputs.size(), device=inputs.device)*bound
        # b = b.to(inputs.device)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """

    def __init__(self,
                 ch,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=.1,
                 reparam_offset=2**-18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = torch.FloatTensor([reparam_offset])

        # Will be updated after the first batch
        self.current_device = 'cpu'

        self.build(ch)

    def build(self, ch):
        self.pedestal = self.reparam_offset**2
        self.beta_bound = (self.beta_min + self.reparam_offset**2)**.5
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch)+self.pedestal)
        self.beta = nn.Parameter(beta)

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init*eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma)
        self.pedestal = self.pedestal

        # Push data to current device
        self.push_to_current_device()

    def push_to_current_device(self):
        self.beta = self.beta.to(self.current_device)
        self.beta_bound = self.beta_bound.to(self.current_device)
        self.gamma = self.gamma.to(self.current_device)
        self.gamma_bound = self.gamma_bound.to(self.current_device)
        self.pedestal = self.pedestal.to(self.current_device)

    def forward(self, inputs):
        # The second condition is only here when reloading from a checkpoint
        # when doing this, the device in current device is the correct one
        # eventhough beta, gamma and so on are not on te current device
        if (inputs.device != self.current_device) or (inputs.device != self.beta_bound.device):
            # Push to a new device only if it has changed
            self.current_device = inputs.device
            self.push_to_current_device()

        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d*w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound().apply(self.beta, self.beta_bound)
        # lower_bound_fn = LowerBound()
        # beta = lower_bound_fn(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound().apply(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs**2, gamma, bias=beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)

        return outputs


class Quantizer(nn.Module):

    def __init__(self):
        super(Quantizer, self).__init__()

    def forward(self, x, fine_tune=False):
        cur_device = x.device
        if self.training or fine_tune:
            res = x + (torch.rand(x.size(), device=cur_device) - 0.5)
        else:
            res = torch.round(x)

        return res


class PdfParamParameterizer(nn.Module):

    def __init__(self, ec_mode, nb_ft):
        super(PdfParamParameterizer, self).__init__()
        self.ec_mode = ec_mode
        # nb_ft: number of feature maps parameterized by x
        self.nb_ft = nb_ft

    def forward(self, x):
        """
        Arguments:
            x: Parameters to be re-parameterize (namely sigma)

        Returns:
            pdf parameters stored as a list of dic
        """
        cur_device = x.device
        if 'two' in self.ec_mode.split('_'):
            K = 2
        elif 'three' in self.ec_mode.split('_'):
            K = 3
        else:
            K = 1

        C = self.nb_ft
        B, _, H, W = x.size()

        # Retrieve all parameters for all the component of the mixture
        # The dimension 1 (K) corresponds to the index of the
        # different components of the gaussian mixture.
        # Mu:
        all_mu = torch.zeros((B, K, C, H, W), device=cur_device)
        start_idx = 0
        for k in range(K):
            all_mu[:, k, :, :, :] = x[:, start_idx: start_idx + C, :, :]
            start_idx += C

        # Sigma. Use of the so-called log var trick to reparemeterize sigma > 0
        all_sigma = torch.zeros((B, K, C, H, W), device=cur_device)
        for k in range(K):
            all_sigma[:, k, :, :, :] = torch.exp(
                0.5 * torch.clamp(
                    x[:, start_idx: start_idx + C, :, :],
                    min=LOG_VAR_MIN,
                    max=LOG_VAR_MAX
                )
            )
            start_idx += C

        # ! Never used
        # Gamma. Use of the so-called log var trick to reparemeterize gamma > 0
        all_gamma = torch.zeros((B, K, C, H, W), device=cur_device)
        flag_gamma = 'gamma' in self.ec_mode.split('_')
        for k in range(K):
            if flag_gamma:
                all_gamma[:, k, :, :, :] = torch.exp(
                    0.5 * torch.clamp(
                        x[:, start_idx: start_idx + C, :, :],
                        min=LOG_VAR_MIN,
                        max=LOG_VAR_MAX
                    )
                )
                start_idx += C
            else:
                all_gamma[:, k, :, :, :] = torch.ones((B, C, H, W), device=cur_device)

        # Weight of each component. We want the sum of weight to be
        # always equals to 1. Thus, we don't need to output K weight but only
        # K - 1 as the last one can be easily deduced. For convenience, we use
        # a softmax to ensure that sum of weight is equal to 1. 
        # Because the softmax needs K inputs, we hard-wire the weight of the 1st
        # component to be 1 before softmax. Then we cat the other K - 1 weight
        # and finally we use the softmax

        # The dimension 1 corresponds to the index of the weight of the
        # different components of the gaussian mixture. We need it to perform
        # the softmax easily
        all_weight = torch.ones((B, K, C, H, W), device=cur_device)

        # Retrieve all the others weights
        for k in range(1, K):
            all_weight[:, k, :, :, :] = x[:, start_idx: start_idx + C, :, :]
            start_idx += C
        norm_all_weight = torch.nn.functional.softmax(all_weight, dim=1)

        pdf_param = []

        for k in range(K):
            pdf_param.append(
                {
                    'mu': all_mu[:, k, :, :, :],
                    'sigma': all_sigma[:, k, :, :, :],
                    'gamma': all_gamma[:, k, :, :, :],
                    'weight': norm_all_weight[:, k, :, :, :],
                }
            )

        return pdf_param
