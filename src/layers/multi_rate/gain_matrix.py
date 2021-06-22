"""
This module re-implements the gain vector used in [1] and described in [2].

They allow to train one system which works at N different rates (i.e. using N
lambdas). Then it is possible to work at any rate by interpolating the
N gain vectors.

    [1] Variable Rate Image Compression with Content Adaptive
        Optimization, Guo et al., CVPR 2020.

    [2] G-VAE: A Continuously Variable Rate Deep Image Compression
        Framework, Cui et al., preprint 2020.
"""

import numpy as np
import torch
from torch.nn import Module, Parameter, ParameterList
from func_util.math_func import xavier_init
from func_util.console_display import print_log_msg
from func_util.nn_util import get_value
from func_util.GOP_structure import FRAME_I

class GainMatrix(Module):

    def __init__(self, param):
        """
        A gain matrix gathers 2N gain vectors :
            - N for the encoder
            - N for the decoder

        It is design to work at N different rates. One gain vector is composed
        of nb_ft elements as it needs to weight nb_ft channels in the
        bottleneck

        """
        super(GainMatrix, self).__init__()

        DEFAULT_PARAM = {
            # Number of different gain vectors
            'N': None,
            # Number of features in the bottleneck in between the gain matrix.
            # Each gain vector has <nb_ft> components
            'nb_ft': None,
            # If true, don't use a random xavier initialization. Instead,
            # gain vector are initialized to 1. This is useful when adding 
            # gain vector to a pre-trained model. Latents start untouched
            # and gain vector don't break anything
            'initialize_to_one': True,
            # If true, use the same scalar gain for all features
            'scalar_gain': False,
        }

        N = get_value('N', param, DEFAULT_PARAM)
        nb_ft = get_value('nb_ft', param, DEFAULT_PARAM)
        initialize_to_one = get_value('initialize_to_one', param, DEFAULT_PARAM)
        scalar_gain = get_value('scalar_gain', param, DEFAULT_PARAM)

        # All the gain vectors at the encoder
        self.enc_gain_list = ParameterList()
        # All the gain vectors at the decoder
        self.dec_gain_list = ParameterList()

        # We add the N gains, although the first one may be 
        # forced to one
        for i in range(N):

            if scalar_gain:
                gain_vec_dim = (1, 1, 1)
            else:
                gain_vec_dim = (nb_ft, 1, 1)

            if initialize_to_one:
                cur_enc_gain = torch.ones(gain_vec_dim, requires_grad=True)
                cur_dec_gain = torch.ones(gain_vec_dim, requires_grad=True)
            else:
                cur_enc_gain = xavier_init(gain_vec_dim)
                cur_dec_gain = xavier_init(gain_vec_dim)

            self.enc_gain_list.append(Parameter(cur_enc_gain))
            self.dec_gain_list.append(Parameter(cur_dec_gain))


    def forward(self, param):
        """
        Perform the scaling / inverse scaling of x using the appropriate gain
        vector.
        """

        DEFAULT_PARAM = {
            # Data to scale or inv. scale
            'x': None,
            # During training, int index of the gain vector to use (depends on
            # the lambda). During inference, float index (i.e. interpolation) of
            # the gain vector to use. If training, idx_rate must be an integer.
            'idx_rate': 0.,
            # Choose between two modes:
            #   <enc>: use enc_gain_list[idx_rate] (to be used at the end of the
            # encoder).
            #   <dec>: use dec_gain_list[idx_rate] (to be used at the decoder
            # input).
            'mode': None,
        }

        x = get_value('x', param, DEFAULT_PARAM)
        idx_rate = get_value('idx_rate', param, DEFAULT_PARAM)
        mode = get_value('mode', param, DEFAULT_PARAM)
        net_out = {}

        # For now we don't interpolate. We just take the gain as it is.
        if self.training:
            cur_gain = self.get_gain(idx_rate, mode=mode)
        else:
            cur_gain = self.interpolate_gain_vector(idx_rate, mode=mode)

        net_out['output'] = x * cur_gain

        return net_out

    def get_gain(self, idx_rate, mode=''):
        """
        Return a specific gain vector, either <enc> or <dec>, for rate index
        <idx_rate>. This function reparemeterize the gain vector to be
        positive by using an absolute value.

        Inputs:
        * <idx_rate> - int
        ?   Index of the gain vector to use.
        ? idx_rate = 0: use the highest rate gain vector

        * <mode> - str
        ?   <enc>: use enc_gain_list[idx_rate] (to be used at the end of the
        ? encoder).
        ?   <dec>: use dec_gain_list[idx_rate] (to be used at the decoder
        ? input).

        Return:
        * <cur_gain.abs()>
        ?   Absolute value of the requested gain
        """
        if mode == 'enc':
            cur_gain = self.enc_gain_list[idx_rate]

        elif mode == 'dec':
            cur_gain = self.dec_gain_list[idx_rate]
        else:
            print_log_msg('ERROR', 'GainMatrix', 'unknown mode', mode)

        return cur_gain.abs()

    def interpolate_gain_vector(self, idx_rate, mode=''):
        """
        Interpolate gain vectors as described in [2] to achieve multi-rate
        inference.

        * <idx_rate> - int or float
        ?   Float index (i.e. interpolation) of the gain vector to use.
        ? idx_rate = 0: use the highest rate gain vector
        ? idx_rate = 0.5: half way between idx_rate = 0 and idx_rate = 1

        * <mode> - str
        ?   <enc>: use enc_gain_list[idx_rate] (to be used at the end of the
        ? encoder).
        ?   <dec>: use dec_gain_list[idx_rate] (to be used at the decoder
        ? input).
        """

        # Notations follow those in [2]
        prev_integer = int(np.floor(idx_rate))
        next_integer = int(np.floor(idx_rate) + 1)

        # l = 1 - distance between idx_rate and the previous integer.
        # When l = 1, we use the gain of the idx_rate (which is an integer)
        # When l = 0, we use gain of the the next idx_rate (also an integer)        
        l = 1 - (idx_rate - prev_integer)

        # For the last value, there is no next_integer because we're already
        # at the last idx rate so we set
        if next_integer == len(self.enc_gain_list):
            next_integer = prev_integer

        m_r = self.get_gain(prev_integer, mode=mode)
        m_t = self.get_gain(next_integer, mode=mode)
        cur_gain = (m_r ** l) * (m_t ** (1 - l))

        return cur_gain
