import torch
import torch.utils.data
from torch import nn
from torch.nn import Module, Conv2d, ReplicationPad2d,\
                     LeakyReLU, Sequential, ReLU, ConvTranspose2d
import numpy as np
import torch.nn.functional as F

from layers.misc.misc_layers import GDN


class ChengResBlock(Module):

    def __init__(self, nb_ft, mode='plain'):
        """
        Reimplementation of the residual blocks defined in [1]

        [1] "Deep Residual Learning for Image Compression", Cheng et al, 2019
        
        * <nb_ft>:
        ?       Number of internal features for all conv. layers

        * <mode>:
        ?       <plain>: standard mode, no upscaling nor downscaling
        ?       <down> : downscaling by 2
        ?       <up_tconv>: upscaling by 2 with a transposed conv
        """

        super(ChengResBlock, self).__init__()

        self.mode = mode
        # In plain mode, non_linearity is Leaky ReLU
        if self.mode == 'plain':
            self.layers = Sequential(
                CustomConvLayer(
                    k_size=3,
                    in_ft=nb_ft,
                    out_ft=nb_ft,
                    non_linearity='leaky_relu'
                    ),
                CustomConvLayer(
                    k_size=3,
                    in_ft=nb_ft,
                    out_ft=nb_ft,
                    non_linearity='leaky_relu'
                    )
                )

        elif self.mode == 'down':
            self.layers = Sequential(
                CustomConvLayer(
                    k_size=3,
                    in_ft=nb_ft,
                    out_ft=nb_ft,
                    non_linearity='leaky_relu',
                    conv_stride=2
                    ),
                CustomConvLayer(
                    k_size=3,
                    in_ft=nb_ft,
                    out_ft=nb_ft,
                    non_linearity='gdn'
                    )
                )

            self.aux_layer = Conv2d(nb_ft, nb_ft, 1, stride=2)

        elif self.mode == 'up_tconv':
            self.layers = Sequential(
                UpscalingLayer(
                    k_size=3,
                    in_ft=nb_ft,
                    out_ft=nb_ft,
                    non_linearity='leaky_relu',
                    mode='transposed'
                    ),
                CustomConvLayer(
                    k_size=3,
                    in_ft=nb_ft,
                    out_ft=nb_ft,
                    non_linearity='gdn_inverse'
                    )
                )

            # ! WARNING: Modification here for a kernel size of 3 instead of 1
            # ! because a transposed conv with k_size = 1 doesn't really make
            # ! sense.
            self.aux_layer = UpscalingLayer(
                    k_size=3,
                    in_ft=nb_ft,
                    out_ft=nb_ft,
                    non_linearity='no',
                    mode='transposed'
                )

    def forward(self, x):
        if self.mode == 'plain':
            return x + self.layers(x)
        elif self.mode == 'down' or self.mode == 'up_tconv':
            return self.aux_layer(x) + self.layers(x)


class ResBlock(Module):

    def __init__(self, k_size, nb_ft):
        super(ResBlock, self).__init__()
        nb_pix_to_pad = int(np.floor(k_size / 2))
        self.layers = Sequential(
            ReplicationPad2d(nb_pix_to_pad),
            Conv2d(nb_ft, nb_ft, k_size),
            ReLU(),
            ReplicationPad2d(nb_pix_to_pad),
            Conv2d(nb_ft, nb_ft, k_size),
        )

    def forward(self, x):
        return F.relu(x + self.layers(x))


class CustomConvLayer(nn.Module):
    """
    Easier way to use convolution. Perform automatically replication pad
    to preserve spatial dimension

    non_linearity is either:
        <gdn>
        <gdn_inverse>
        <leaky_relu>
        <relu>
        <no>
    """
    def __init__(self, k_size=5, in_ft=64, out_ft=64, flag_bias=True,
                 non_linearity='leaky_relu', conv_stride=1,
                 padding_mode='replicate'):
        super(CustomConvLayer, self).__init__()
        nb_pix_to_pad = int(np.floor(k_size / 2))

        if padding_mode == 'replicate':
            padding_fn = ReplicationPad2d(nb_pix_to_pad)

        self.layers = Sequential(
            padding_fn,
            Conv2d(in_ft, out_ft, k_size, stride=conv_stride, bias=flag_bias)
        )

        if non_linearity == 'gdn':
            self.layers.add_module(
                'non_linearity',
                GDN(out_ft, inverse=False)
            )

        elif non_linearity == 'gdn_inverse':
            self.layers.add_module(
                'non_linearity',
                GDN(out_ft, inverse=True)
            )

        elif non_linearity == 'leaky_relu':
            self.layers.add_module(
                'non_linearity',
                LeakyReLU()
            )

        elif non_linearity == 'relu':
            self.layers.add_module(
                'non_linearity',
                ReLU()
            )

    def forward(self, x):
        return self.layers(x)


class UpscalingLayer(nn.Module):
    def __init__(self, k_size=5, in_ft=64, out_ft=64, flag_bias=True,
                 non_linearity='leaky_relu', mode='transposed',
                 flag_first_layer=False):
        """
        Upscaling with a factor of two

        * <non_linearity>:
        ?   gdn, gdn_inverse, leaky_relu, relu, no

        * <mode>:
        ?   transposed:
        ?       Use a transposed conv to perform upsampling
        ?   transposedtransposed_no_bias:
        ?       Use a transposed conv to perform upsampling, without a bias

        """

        super(UpscalingLayer, self).__init__()

        # Transposed conv param, computed thanks to
        # https://pytorch.org/docs/stable/nn.html#convtranspose2d
        # t_dilat = 1 # defaut value
        t_stride = 2
        t_outpad = 1
        t_padding = int(((t_outpad + k_size) / 2) - 1)

        # Override flag_bias
        if mode == 'transposed_no_bias':
            flag_bias = False

        self.layers = Sequential(
            ConvTranspose2d(
                    in_ft,
                    out_ft,
                    k_size,
                    stride=t_stride,
                    padding=t_padding,
                    output_padding=t_outpad,
                    bias=flag_bias
                )
        )

        # Add the correct non-linearity
        if non_linearity == 'gdn':
            self.layers.add_module(
                'non_linearity',
                GDN(out_ft, inverse=False)
            )

        elif non_linearity == 'gdn_inverse':
            self.layers.add_module(
                'non_linearity',
                GDN(out_ft, inverse=True)
            )

        elif non_linearity == 'leaky_relu':
            self.layers.add_module(
                'non_linearity',
                LeakyReLU()
            )

        elif non_linearity == 'relu':
            self.layers.add_module(
                'non_linearity',
                ReLU()
            )


    def forward(self, x):
        return self.layers(x)
