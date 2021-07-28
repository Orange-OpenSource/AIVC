# Software Name: AIVC
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

"""
This model gathers building blocks (encoder/decoder main/hp) of different
well-known compression auto-encoder. They are used to construct a conditional
net module to have a generic AE architecture.
"""

from torch.nn import ModuleList, Module, Sequential

from layers.misc.custom_conv_layers import UpscalingLayer, CustomConvLayer,\
                                           ChengResBlock
from func_util.nn_util import get_value
from layers.misc.attention import SimplifiedAttention

"""
My (custom) baseline model. It is centered around the standard Minnen's
model [1] without the auto-regressive component.
To improve performance, I used lightweight residual blocks from [2, 3]

[1] "Joint Autoregressive and Hierarchical Priors for Learned Image
Compression", Minnen et al., NIPS 2018

[2] "Learned Image Compression with Discretized Gaussian Mixture
Likelihoods and Attention Modules", Cheng et al, 2020

[3] "Deep Residual Learning for Image Compression", Cheng et al, 2019
"""


class BaseEncoder(Module):

    def __init__(self, param):

        super(BaseEncoder, self).__init__()

        self.DEFAULT_PARAM = {
            # Number of input features
            'in_c': 3,
            # Number of output features (in the bottleneck)
            'out_c': 256,
            # Number of internal features for all conv. layers
            'nb_ft': 64,
            # Type of attention module used.
            # ? no
            #       No attention module.
            # ? cheng
            #       Attention module from "Learned Image Compression with
            #       Discretized Gaussian Mixture Likelihoods and Attention
            #       Modules", Cheng et al.
            # ? cheng14
            #       Only the first and the last attention module of "cheng"
            #       option (The ones the furthest from the bottleneck).
            # ? cheng24
            #       Only the second and lath attention modules of "cheng".
            #       Last of the encoder and last of the decoder
            # ? X_lightweight
            #       X is the three options aboved. Same thing but use
            #       the lightweight resblock in the attention module.
            'attention': 'cheng14_lightweight',
            # Kernel size for all convolution layers, except attention modules.
            'k_size': 5,
            # Which non-linearity is used after the convolution layers. This
            # is not used for the attention module.
            # ? gdn
            # ? gdn_inverse
            # ? relu
            # ? leaky_relu
            # ? no
            'nl_type': 'gdn',
            # Do we use a bias for the last layer:
            'flag_bias_last_layer': False,
        }

        in_c = get_value('in_c', param, self.DEFAULT_PARAM)
        out_c = get_value('out_c', param, self.DEFAULT_PARAM)
        nb_ft = get_value('nb_ft', param, self.DEFAULT_PARAM)
        attention = get_value('attention', param, self.DEFAULT_PARAM)
        k_size = get_value('k_size', param, self.DEFAULT_PARAM)
        nl_type = get_value('nl_type', param, self.DEFAULT_PARAM)
        flag_bias_last_layer = get_value(
            'flag_bias_last_layer', param, self.DEFAULT_PARAM
        )

        self.layers = ModuleList()

        self.layers.append(
            CustomConvLayer(
                k_size=k_size,
                in_ft=in_c,
                out_ft=nb_ft,
                flag_bias=True,
                non_linearity=nl_type,
                conv_stride=2,
            )
        )

        self.layers.append(
            CustomConvLayer(
                k_size=k_size,
                in_ft=nb_ft,
                out_ft=nb_ft,
                flag_bias=True,
                non_linearity=nl_type,
                conv_stride=2,
            )
        )

        # Deal with attention module in position 1.
        if ('cheng' in attention.split('_')) or\
           ('cheng14' in attention.split('_')):
            flag_attention = True
        else:
            flag_attention = False

        if flag_attention:
            if 'lightweight' in attention.split('_'):
                flag_lightweight_resb = True
            else:
                flag_lightweight_resb = False

            self.layers.append(
                SimplifiedAttention(
                    nb_ft,
                    lightweight_resblock=flag_lightweight_resb
                )
            )

        self.layers.append(
            CustomConvLayer(
                k_size=k_size,
                in_ft=nb_ft,
                out_ft=nb_ft,
                flag_bias=True,
                non_linearity=nl_type,
                conv_stride=2,
            )
        )

        self.layers.append(
            CustomConvLayer(
                k_size=k_size,
                in_ft=nb_ft,
                out_ft=out_c,
                flag_bias=flag_bias_last_layer,
                non_linearity='no',
                conv_stride=2,
            )
        )

        # Deal with attention module in position 2.
        if ('cheng' in attention.split('_')) or\
           ('cheng24' in attention.split('_')):
            flag_attention = True
        else:
            flag_attention = False

        if flag_attention:
            if 'lightweight' in attention.split('_'):
                flag_lightweight_resb = True
            else:
                flag_lightweight_resb = False

            self.layers.append(
                SimplifiedAttention(
                    out_c,
                    lightweight_resblock=flag_lightweight_resb
                )
            )

    def forward(self, x):
        for lay in self.layers:
            x = lay(x)
        return x


class BaseDecoder(Module):

    def __init__(self, param):

        super(BaseDecoder, self).__init__()

        self.DEFAULT_PARAM = {
            # Number of input features (in the bottleneck)
            'in_c': 256,
            # Number of output features
            'out_c': 3,
            # Number of internal features for all conv. layers
            'nb_ft': 64,
            # Type of attention module used.
            # ? no
            #       No attention module.
            # ? cheng
            #       Attention module from "Learned Image Compression with
            #       Discretized Gaussian Mixture Likelihoods and Attention
            #       Modules", Cheng et al.
            # ? cheng14
            #       Only the first and the last attention module of "cheng"
            #       option (The ones the furthest from the bottleneck).
            # ? cheng24
            #       Only the second and lath attention modules of "cheng".
            #       Last of the encoder and last of the decoder
            # ? X_lightweight
            #       X is the three options aboved. Same thing but use
            #       the lightweight resblock in the attention module.
            'attention': 'cheng14_lightweight',
            # Kernel size for all convolution layers, except attention modules.
            'k_size': 5,
            # Which non-linearity is used after the convolution layers. This
            # is not used for the attention module.
            # ? gdn
            # ? gdn_inverse
            # ? relu
            # ? leaky_relu
            # ? no
            'nl_type': 'gdn_inverse',
        }

        in_c = get_value('in_c', param, self.DEFAULT_PARAM)
        out_c = get_value('out_c', param, self.DEFAULT_PARAM)
        nb_ft = get_value('nb_ft', param, self.DEFAULT_PARAM)
        attention = get_value('attention', param, self.DEFAULT_PARAM)
        k_size = get_value('k_size', param, self.DEFAULT_PARAM)
        nl_type = get_value('nl_type', param, self.DEFAULT_PARAM)

        self.layers = ModuleList()

        # Deal with attention module in position 1.
        if ('cheng' in attention.split('_')):
            flag_attention = True
        else:
            flag_attention = False

        if flag_attention:
            if 'lightweight' in attention.split('_'):
                flag_lightweight_resb = True
            else:
                flag_lightweight_resb = False

            self.layers.append(
                SimplifiedAttention(
                    in_c,
                    lightweight_resblock=flag_lightweight_resb
                )
            )

        self.layers.append(
            UpscalingLayer(
                k_size=k_size,
                in_ft=in_c,
                out_ft=nb_ft,
                flag_bias=True,
                non_linearity=nl_type,
            )
        )

        self.layers.append(
            UpscalingLayer(
                k_size=k_size,
                in_ft=nb_ft,
                out_ft=nb_ft,
                flag_bias=True,
                non_linearity=nl_type,
            )
        )

        # Deal with attention module in position 1.
        if ('cheng' in attention.split('_')) or\
           ('cheng14' in attention.split('_')) or\
           ('cheng24' in attention.split('_')):
            flag_attention = True
        else:
            flag_attention = False

        if flag_attention:
            if 'lightweight' in attention.split('_'):
                flag_lightweight_resb = True
            else:
                flag_lightweight_resb = False

            self.layers.append(
                SimplifiedAttention(
                    nb_ft,
                    lightweight_resblock=flag_lightweight_resb
                )
            )

        self.layers.append(
            UpscalingLayer(
                k_size=k_size,
                in_ft=nb_ft,
                out_ft=nb_ft,
                flag_bias=True,
                non_linearity=nl_type,
            )
        )

        self.layers.append(
            UpscalingLayer(
                k_size=k_size,
                in_ft=nb_ft,
                out_ft=out_c,
                flag_bias=True,
                non_linearity='no',
            )
        )

    def forward(self, x):
        for lay in self.layers:
            x = lay(x)
        return x


class BaseEncoderHP(Module):
    def __init__(self, param):
        super(BaseEncoderHP, self).__init__()

        self.DEFAULT_PARAM = {
            # Number of input features
            'in_c': 256,
            # Number of output features (in the bottleneck)
            'out_c': 64,
            # Number of internal features for all conv. layers
            'nb_ft': 64,
            # Kernel size for all convolution layers, except attention modules.
            'k_size': 5,
            # Which non-linearity is used after the convolution layers. This
            # is not used for the attention module.
            # ? gdn
            # ? gdn_inverse
            # ? relu
            # ? leaky_relu
            # ? no
            'nl_type': 'leaky_relu',
            # Do we use a bias for the last layer:
            'flag_bias_last_layer': False,
        }

        in_c = get_value('in_c', param, self.DEFAULT_PARAM)
        out_c = get_value('out_c', param, self.DEFAULT_PARAM)
        nb_ft = get_value('nb_ft', param, self.DEFAULT_PARAM)
        k_size = get_value('k_size', param, self.DEFAULT_PARAM)
        nl_type = get_value('nl_type', param, self.DEFAULT_PARAM)
        flag_bias_last_layer = get_value(
            'flag_bias_last_layer', param, self.DEFAULT_PARAM
        )

        self.layers = ModuleList()

        self.layers.append(
            CustomConvLayer(
                k_size=k_size,
                in_ft=in_c,
                out_ft=nb_ft,
                flag_bias=True,
                non_linearity=nl_type,
                conv_stride=2,
            )
        )

        self.layers.append(
            CustomConvLayer(
                k_size=k_size,
                in_ft=nb_ft,
                out_ft=nb_ft,
                flag_bias=True,
                non_linearity=nl_type,
                conv_stride=2,
            )
        )

        self.layers.append(
            CustomConvLayer(
                k_size=k_size,
                in_ft=nb_ft,
                out_ft=out_c,
                flag_bias=flag_bias_last_layer,
                non_linearity='no',
                conv_stride=1,
            )
        )

    def forward(self, x):
        for lay in self.layers:
            x = lay(x)
        return x


class BaseDecoderHP(Module):
    def __init__(self, param):
        super(BaseDecoderHP, self).__init__()

        self.DEFAULT_PARAM = {
            # Number of input features (in the bottleneck)
            'in_c': 64,
            # Number of output features
            'out_c': 512,
            # Number of internal features for all conv. layers
            'nb_ft': 64,
            # Kernel size for all convolution layers, except attention modules.
            'k_size': 5,
            # Which non-linearity is used after the convolution layers. This
            # is not used for the attention module.
            # ? gdn
            # ? gdn_inverse
            # ? relu
            # ? leaky_relu
            # ? no
            'nl_type': 'leaky_relu',
        }

        in_c = get_value('in_c', param, self.DEFAULT_PARAM)
        out_c = get_value('out_c', param, self.DEFAULT_PARAM)
        nb_ft = get_value('nb_ft', param, self.DEFAULT_PARAM)
        k_size = get_value('k_size', param, self.DEFAULT_PARAM)
        nl_type = get_value('nl_type', param, self.DEFAULT_PARAM)

        self.layers = ModuleList()

        self.layers.append(
            CustomConvLayer(
                k_size=k_size,
                in_ft=in_c,
                out_ft=nb_ft,
                flag_bias=True,
                non_linearity=nl_type,
                conv_stride=1,
            )
        )

        self.layers.append(
            UpscalingLayer(
                k_size=k_size,
                in_ft=nb_ft,
                out_ft=nb_ft,
                flag_bias=True,
                non_linearity=nl_type,
            )
        )

        self.layers.append(
            UpscalingLayer(
                k_size=k_size,
                in_ft=nb_ft,
                out_ft=out_c,
                flag_bias=True,
                non_linearity='no',
            )
        )

    def forward(self, x):
        for lay in self.layers:
            x = lay(x)
        return x


"""
Cheng model is a re-implementation of the model described in [1] with the help 
from  info in [2], particularly for the residual blocks.

[1] "Learned Image Compression with Discretized Gaussian Mixture
Likelihoods and Attention Modules", Cheng et al, 2020

[2] "Deep Residual Learning for Image Compression", Cheng et al, 2019
"""


class ChengEncoder(Module):

    def __init__(self, param):
        super(ChengEncoder, self).__init__()

        self.DEFAULT_PARAM = {
            # Number of input features
            'in_c': 3,
            # Number of output features (in the bottleneck)
            'out_c': 256,
            # Number of internal features for all conv. layers
            'nb_ft': 64,
            # Type of attention module used.
            # ? no
            #       No attention module.
            # ? cheng
            #       Attention module from "Learned Image Compression with
            #       Discretized Gaussian Mixture Likelihoods and Attention
            #       Modules", Cheng et al.
            # ? cheng14
            #       Only the first and the last attention module of "cheng"
            #       option (The ones the furthest from the bottleneck).
            # ? cheng24
            #       Only the second and lath attention modules of "cheng".
            #       Last of the encoder and last of the decoder
            # ? X_lightweight
            #       X is the three options aboved. Same thing but use
            #       the lightweight resblock in the attention module.
            'attention': 'cheng_lightweight',
            # Do we use a bias for the last layer:
            'flag_bias_last_layer': False,
        }

        in_c = get_value('in_c', param, self.DEFAULT_PARAM)
        out_c = get_value('out_c', param, self.DEFAULT_PARAM)
        nb_ft = get_value('nb_ft', param, self.DEFAULT_PARAM)
        attention = get_value('attention', param, self.DEFAULT_PARAM)
        flag_bias_last_layer = get_value(
            'flag_bias_last_layer', param, self.DEFAULT_PARAM
        )

        self.layers = ModuleList()

        # One small conv layer to go from 3 to nb_ft features
        # ! This is  an invention of mine!

        self.layers.append(
            CustomConvLayer(
                k_size=3,
                in_ft=in_c,
                out_ft=nb_ft,
                non_linearity='gdn'
            )
        )
        self.layers.append(ChengResBlock(nb_ft, mode='down'))
        self.layers.append(ChengResBlock(nb_ft, mode='plain'))
        self.layers.append(ChengResBlock(nb_ft, mode='down'))

        if ('cheng' in attention.split('_')) or\
           ('cheng14' in attention.split('_')):
            if 'lightweight' in attention.split('_'):
                flag_lightweight_resb = True
            else:
                flag_lightweight_resb = False

            self.layers.append(
                SimplifiedAttention(
                    nb_ft,
                    lightweight_resblock=flag_lightweight_resb
                    )
            )

        self.layers.append(ChengResBlock(nb_ft, mode='plain'))
        self.layers.append(ChengResBlock(nb_ft, mode='down'))
        self.layers.append(ChengResBlock(nb_ft, mode='plain'))
        self.layers.append(
            CustomConvLayer(
                    k_size=3,
                    in_ft=nb_ft,
                    out_ft=out_c,
                    non_linearity='no',
                    conv_stride=2,
                    flag_bias=flag_bias_last_layer,
            )
        )

        if ('cheng' in attention.split('_')) or\
           ('cheng24' in attention.split('_')):
            if 'lightweight' in attention.split('_'):
                flag_lightweight_resb = True
            else:
                flag_lightweight_resb = False

            self.layers.append(
                SimplifiedAttention(
                    out_c,
                    lightweight_resblock=flag_lightweight_resb
                    )
            )

    def forward(self, x):
        for lay in self.layers:
            x = lay(x)
        return x


class ChengDecoder(Module):

    def __init__(self, param):

        super(ChengDecoder, self).__init__()

        self.DEFAULT_PARAM = {
            # Number of input features
            'in_c': 256,
            # Number of output features (in the bottleneck)
            'out_c': 3,
            # Number of internal features for all conv. layers
            'nb_ft': 64,
            # Type of attention module used.
            # ? no
            #       No attention module.
            # ? cheng
            #       Attention module from "Learned Image Compression with
            #       Discretized Gaussian Mixture Likelihoods and Attention
            #       Modules", Cheng et al.
            # ? cheng14
            #       Only the first and the last attention module of "cheng"
            #       option (The ones the furthest from the bottleneck).
            # ? cheng24
            #       Only the second and lath attention modules of "cheng".
            #       Last of the encoder and last of the decoder
            # ? X_lightweight
            #       X is the three options aboved. Same thing but use
            #       the lightweight resblock in the attention module.
            'attention': 'cheng_lightweight',
        }

        in_c = get_value('in_c', param, self.DEFAULT_PARAM)
        out_c = get_value('out_c', param, self.DEFAULT_PARAM)
        nb_ft = get_value('nb_ft', param, self.DEFAULT_PARAM)
        attention = get_value('attention', param, self.DEFAULT_PARAM)

        self.layers = ModuleList()

        if 'cheng' in attention.split('_'):
            if 'lightweight' in attention.split('_'):
                flag_lightweight_resb = True
            else:
                flag_lightweight_resb = False

            self.layers.append(
                SimplifiedAttention(
                    in_c,
                    lightweight_resblock=flag_lightweight_resb
                    )
            )

        # One small conv layer to go from in_c to nb_ft features
        # ! This is  an invention of mine!
        self.layers.append(
            CustomConvLayer(
                k_size=3,
                in_ft=in_c,
                out_ft=nb_ft,
                non_linearity='gdn'
            )
        )

        self.layers.append(ChengResBlock(nb_ft, mode='plain'))
        self.layers.append(ChengResBlock(nb_ft, mode='up_tconv'))
        self.layers.append(ChengResBlock(nb_ft, mode='plain'))
        self.layers.append(ChengResBlock(nb_ft, mode='up_tconv'))

        if ('cheng' in attention.split('_')) or\
           ('cheng24' in attention.split('_')) or\
           ('cheng14' in attention.split('_')):
            if 'lightweight' in attention.split('_'):
                flag_lightweight_resb = True
            else:
                flag_lightweight_resb = False

            self.layers.append(
                SimplifiedAttention(
                    nb_ft,
                    lightweight_resblock=flag_lightweight_resb
                    )
            )

        self.layers.append(ChengResBlock(nb_ft, mode='plain'))
        self.layers.append(ChengResBlock(nb_ft, mode='up_tconv'))
        self.layers.append(ChengResBlock(nb_ft, mode='plain'))
        self.layers.append(
                UpscalingLayer(
                    k_size=3,
                    in_ft=nb_ft,
                    out_ft=out_c,
                    non_linearity='no',
                    mode='transposed'
                    ),
        )

    def forward(self, x):
        for lay in self.layers:
            x = lay(x)
        return x


class ChengEncoderHp(Module):

    def __init__(self, param):
        super(ChengEncoderHp, self).__init__()

        self.DEFAULT_PARAM = {
            # Number of input features
            'in_c': 256,
            # Number of output features (in the bottleneck)
            'out_c': 64,
            # Number of internal features for all conv. layers
            'nb_ft': 64,
            # Do we use a bias for the last layer:
            'flag_bias_last_layer': False,
        }
        in_c = get_value('in_c', param, self.DEFAULT_PARAM)
        out_c = get_value('out_c', param, self.DEFAULT_PARAM)
        nb_ft = get_value('nb_ft', param, self.DEFAULT_PARAM)
        flag_bias_last_layer = get_value(
            'flag_bias_last_layer', param, self.DEFAULT_PARAM
        )

        self.layers = Sequential(
            CustomConvLayer(
                k_size=3,
                in_ft=in_c,
                out_ft=nb_ft,
                non_linearity='leaky_relu'
            ),
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
                non_linearity='leaky_relu',
                conv_stride=2
            ),
            CustomConvLayer(
                k_size=3,
                in_ft=nb_ft,
                out_ft=nb_ft,
                non_linearity='leaky_relu'
            ),
            CustomConvLayer(
                k_size=3,
                in_ft=nb_ft,
                out_ft=out_c,
                non_linearity='no',
                conv_stride=2,
                flag_bias=flag_bias_last_layer
            )
        )

    def forward(self, x):
        return self.layers(x)


class ChengDecoderHp(Module):

    def __init__(self, param):
        super(ChengDecoderHp, self).__init__()

        self.DEFAULT_PARAM = {
            # Number of input features
            'in_c': 64,
            # Number of output features (in the bottleneck)
            'out_c': 512,
            # Number of internal features for all conv. layers
            'nb_ft': 64,
        }
        in_c = get_value('in_c', param, self.DEFAULT_PARAM)
        out_c = get_value('out_c', param, self.DEFAULT_PARAM)
        nb_ft = get_value('nb_ft', param, self.DEFAULT_PARAM)

        self.layers = Sequential(
            CustomConvLayer(
                k_size=3,
                in_ft=in_c,
                out_ft=nb_ft,
                non_linearity='leaky_relu'
            ),
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
                out_ft=int(1.5 * nb_ft),
                non_linearity='leaky_relu',
            ),
            UpscalingLayer(
                k_size=3,
                in_ft=int(1.5 * nb_ft),
                out_ft=int(1.5 * nb_ft),
                non_linearity='leaky_relu',
                mode='transposed'
            ),
            CustomConvLayer(
                k_size=3,
                in_ft=int(1.5 * nb_ft),
                out_ft=int(2 * nb_ft),
                non_linearity='leaky_relu'
            ),
            # Replicate the fusion module eventhough we don't have
            # anything to merge
            CustomConvLayer(
                k_size=1,
                in_ft=int(2 * nb_ft),
                out_ft=out_c,
                non_linearity='leaky_relu'
            ),
            CustomConvLayer(
                k_size=1,
                in_ft=out_c,
                out_ft=out_c,
                non_linearity='leaky_relu'
            ),
            CustomConvLayer(
                k_size=1,
                in_ft=out_c,
                out_ft=out_c,
                non_linearity='no'
            ),
        )

    def forward(self, x):
        return self.layers(x)
