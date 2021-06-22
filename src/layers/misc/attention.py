"""
    Local attention module used in [1]

[1]  "Learned Image Compression with Discretized Gaussian Mixture Likelihoods
and Attention Modules", Cheng et al.
"""
from torch.nn import Module, Conv2d, Sequential, Sigmoid, ReplicationPad2d,\
                     ConvTranspose2d, LeakyReLU
from layers.misc.custom_conv_layers import ResBlock
import torch.nn.functional as F


class AttentionResBlock(Module):
    """
    From [3]
    """

    def __init__(self, nb_ft):
        super(AttentionResBlock, self).__init__()

        half_ft = int(nb_ft / 2)

        self.layers = Sequential(
            Conv2d(nb_ft, half_ft, 1),
            LeakyReLU(),
            ReplicationPad2d(1),
            Conv2d(half_ft, half_ft, 3),
            LeakyReLU(),
            Conv2d(half_ft, nb_ft, 1),
        )

    def forward(self, x):
        return F.leaky_relu(x + self.layers(x))


class SimplifiedAttention(Module):
    """
    From [3]
    """

    def __init__(self, nb_ft, k_size=3, lightweight_resblock=False):
        super(SimplifiedAttention, self).__init__()

        self.nb_ft = nb_ft
        self.k_size = k_size

        # In [3], the res block does not always operate at nb_ft, we denote
        # this option as light_resb, opposite to standard resblock operating
        # at full nb_ft everything
        # Two networks in this module: the trunk and the attention

        if lightweight_resblock:
            self.trunk = Sequential(
                AttentionResBlock(self.nb_ft),
                AttentionResBlock(self.nb_ft),
                AttentionResBlock(self.nb_ft)
            )

            self.attention = Sequential(
                AttentionResBlock(self.nb_ft),
                AttentionResBlock(self.nb_ft),
                AttentionResBlock(self.nb_ft),
                Conv2d(self.nb_ft, self.nb_ft, 1),
                Sigmoid()
            )

        else:
            self.trunk = Sequential(
                ResBlock(self.k_size, self.nb_ft),
                ResBlock(self.k_size, self.nb_ft),
                ResBlock(self.k_size, self.nb_ft)
            )

            self.attention = Sequential(
                ResBlock(self.k_size, self.nb_ft),
                ResBlock(self.k_size, self.nb_ft),
                ResBlock(self.k_size, self.nb_ft),
                Conv2d(self.nb_ft, self.nb_ft, 1),
                Sigmoid()
            )

    def forward(self, x):
        trunk_out = self.trunk(x)
        attention_out = self.attention(x)

        weighted_trunk = trunk_out * attention_out
        res = weighted_trunk + x
        return res

