# Software Name: AIVC
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

import torchac
import torch
import math
import os
import numpy as np
import scipy.stats

from torch.distributions import Laplace
from func_util.nn_util import get_value
from real_life.utils import BITSTREAM_SUFFIX
from real_life.check_md5sum import write_md5sum, read_md5sum, compute_md5sum

"""
Bitstream structure for each frame:
-----------------------------------

    1. MOFNet z latents

    [Number of bytes written  (4 bytes)] *
    [md5sum of z_m           (32 bytes)] < only if flag_md5sum
    [z entropy coded        (variables)]

    2. MOFNet y latents

    [Number of bytes written  (4 bytes)] *
    [md5sum of y_m           (32 bytes)] < only if flag_md5sum
    [Nb. non zero ft. maps     (1 byte)]
    [Index non zero ft. maps (variable)]
    [y entropy coded        (variables)]

    3. CodecNet z latents
    [Number of bytes written  (4 bytes)] *
    [md5sum of z_c           (32 bytes)] < only if flag_md5sum
    [z entropy coded        (variables)]

    4. CodecNet y latents

    [Number of bytes written  (4 bytes)] *
    [md5sum of y_c           (32 bytes)] < only if flag_md5sum
    [Nb. non zero ft. maps     (1 byte)]
    [Index non zero ft. maps (variable)]
    [y entropy coded        (variables)]

Number of bytes written represents the number of bytes for this subpart of the bitstream,
**excepted** the 4 bytes of number of bytes written. E.g. if we have z entropy coded = 256 bytes,
the number of bytes written is 256, **not 260**
"""


class ArithmeticCoder():

    def __init__(self, param):
        """
        Remark: for all the comments, [] means that the interval bounds are
        included and ][ means that they are excluded.
        """

        DEFAULT_PARAM = {
            # The BallePdfEstim module to compute the CDF
            'balle_pdf_estim_z': None,
            # On which device the code will run
            'device': 'cpu',
            # No value can be outside [-AC_MAX_VAL, AC_MAX_VAL]     
            'AC_MAX_VAL': 256,
        }

        self.balle_pdf_estim = get_value('balle_pdf_estim_z', param, DEFAULT_PARAM)
        device = get_value('device', param, DEFAULT_PARAM)
        
        self.AC_MAX_VAL = get_value('AC_MAX_VAL', param, DEFAULT_PARAM)
        self.pre_computed_z_cdf = self._precompute_z_cdf({'device': device})

    def _precompute_z_cdf(self, param):
        """
        Pre-compute (at the encoder side) the CDF of z for indices in 
        [-AC_MAX_VAL, AC_MAX_VAL - 1]. This is done once for all and then
        used to perform arithmetic coding, decoding.
        """

        DEFAULT_PARAM = {
            # On which device the code will run
            'device': 'cpu',
        }

        device = get_value('device', param, DEFAULT_PARAM)

        nb_ft_z = self.balle_pdf_estim.nb_channel

        # According to the torchac documentation, the symbols sent with entropy
        # coding are in the range [0, Lp - 2]. We have 2 * max_val value to
        # transmit, so we want: Lp - 2 = 2 * max_val
        Lp = 2 * self.AC_MAX_VAL + 2

        # We compute the CDF for all this indices
        # idx are in [-AC_MAX_VAL - 0.5, AC_MAX_VAL + 1.5]
        idx = torch.arange(Lp, device=device).float() - self.AC_MAX_VAL - 0.5

        # It is slightly different than the laplace mode, because the balle pmf
        # only accepts 4D inputs with the last dimension equals to one. Thus,
        # we consider idx as a [1, 1, -1, 1] tensor.
        idx = idx.view(1, 1, -1, 1)
                
        # Because the cumulative are the same for a given feature map,
        # we can spare some computation by just computing them once
        # per feature map. We'll replicate the <ouput_cdf> variables
        # accross dimensions B, H and W according to what we have to transmit
        idx = idx.repeat(1, nb_ft_z, 1, 1)

        # Compute cdf and add back the W channel
        output_cdf = self.balle_pdf_estim.cdf(idx).squeeze(-1).unsqueeze(-2).unsqueeze(-3)

        # Quantize the CDF value to a smaller precision to deal with float
        # unaccuracy. # ! Not needed anymore with the pre-computation!
        # output_cdf = torch.round(output_cdf * 1024).to(torch.int) / 1024.        

        return output_cdf

    def get_y_cdf(self, sigma):
        cur_device = sigma.device
        B, C, H, W = sigma.size()

        # According to the torchac documentation, the symbols sent with entropy
        # coding are in the range [0, Lp - 2]. We have 2 * max_val value to
        # transmit, so we want: Lp - 2 = 2 * max_val
        Lp = 2 * self.AC_MAX_VAL + 2

        # We compute the CDF for all this indices
        # idx are in [-AC_MAX_VAL - 0.5, AC_MAX_VAL + 1.5]
        idx = torch.arange(Lp, device=cur_device).float() - self.AC_MAX_VAL - 0.5

        # Add a 5th dimension to mu and sigma
        sigma = sigma.unsqueeze(-1)

        # Compute the scale parameter
        b = sigma / torch.sqrt(torch.tensor([2.0], device=cur_device))
        # Centered distribution
        mu = torch.zeros_like(b, device=cur_device)
        # Get the distribution
        my_pdf = Laplace(mu, b)

        # Compute cdf
        idx = idx.view(1, 1, 1, 1, -1).repeat(B, C, H, W, 1)
        output_cdf = my_pdf.cdf(idx)

        return output_cdf

    def compute_cdf(self, param):
        """
        Compute the CDF value of the different symbols according to a
        given distribution.
        """

        DEFAULT_PARAM = {
            # Mode is either <laplace> or <pmf>
            'mode': 'laplace',
            # If mode == 'laplace', we need the sigma parameter
            'sigma': None,
            # Encoding with the PMF requires the data dimension (z.size())
            'data_dimension': None,
        }

        mode = get_value('mode', param, DEFAULT_PARAM)
        sigma = get_value('sigma', param, DEFAULT_PARAM)
        data_dimension = get_value('data_dimension', param, DEFAULT_PARAM)

        if mode == 'pmf':
            B, C, H, W = data_dimension
            # Add the spatial dimension to the pre-computed cdf
            output_cdf = self.pre_computed_z_cdf.repeat(B, 1, H, W, 1)

        elif mode == 'laplace':
            # Compute the (quantized) scale parameter floating point sigma
            output_cdf = self.get_y_cdf(sigma)
        
        return output_cdf

    def encode(self, param):

        DEFAULT_PARAM = {
            # The 4-dimensional tensor to encode
            'x': None,
            # Mode is either <laplace> or <pmf>
            'mode': 'laplace',
            # If mode == 'laplace', we need the parameters mu and sigma
            # 'mu': None,
            'sigma': None,
            # If mode == 'pmf', we need the BallePdfEstim module to compute
            # the CDF
            # 'balle_pdf_estim': None,
            # Name (absolute path) of the bitstream
            'bitstream_path': None,
            # Debug. If true, print specific stuff
            'flag_debug': True,
            # Specify what quantity we're entropy coding.
            # It can be <mofnet_y>, <mofnet_z>, <codecnet_y>, codecnet_z>.
            # This is needed for the decoding part, to know which part of the
            # bitstream we should read.
            # Bitstream structure for each frame is detailed above.
            'latent_name': '',
            # Compute a feature-wise md5sum by saving it in a temporary file.
            # Include this md5sum in the bitstream
            'flag_md5sum': False,
        }

        x = get_value('x', param, DEFAULT_PARAM)
        mode = get_value('mode', param, DEFAULT_PARAM)
        sigma = get_value('sigma', param, DEFAULT_PARAM)
        # balle_pdf_estim = get_value('balle_pdf_estim', param, DEFAULT_PARAM)
        bitstream_path = get_value('bitstream_path', param, DEFAULT_PARAM)
        flag_debug = get_value('flag_debug', param, DEFAULT_PARAM)
        latent_name = get_value('latent_name', param, DEFAULT_PARAM)
        flag_md5sum = get_value('flag_md5sum', param, DEFAULT_PARAM)
        
        if not(bitstream_path.endswith(BITSTREAM_SUFFIX)):
            bitstream_path += BITSTREAM_SUFFIX

        # Gather here the bytes to be written in the binary file
        byte_to_write = b''

        if flag_md5sum:
            x_encoder = x.to('cpu').to(torch.int16).numpy().astype(int)
            np.savetxt( './tmp_tensor.npy', x_encoder.flatten())
            encoder_md5sum = compute_md5sum({'in_file': './tmp_tensor.npy'}).encode()
            byte_to_write += encoder_md5sum
            os.system('rm ./tmp_tensor.npy')

        # We don't want to encode empty y feature maps
        if mode == 'laplace':
            # Batch size of x is 1 so we collapse this dimension and we compute the sum
            # of all feature maps.
            # 1-D tensor with the sum of the C feature maps of x
            sum_per_ft = x.abs().sum(dim=(2, 3)).squeeze(0)

            # Should be != 0
            idx_non_zero = sum_per_ft != 0
            x_non_zero = x[:, idx_non_zero, :, :]
            sigma_non_zero = sigma[:, idx_non_zero, :, :]
            nb_ft_map_sent = x_non_zero.size()[1]

            # Write the number of feature map sent and their index in the
            # bitstream file. Everything is written on 1 byte.
            byte_to_write += nb_ft_map_sent.to_bytes(1, byteorder='big')
            for i in range(sum_per_ft.size()[0]):
                # Should be != 0
                if sum_per_ft[i] != 0:
                    byte_to_write += i.to_bytes(1, byteorder='big')

        else:
            sigma_non_zero = sigma
            x_non_zero = x
            nb_ft_map_sent = x.size()[1]

        header_overhead = len(byte_to_write)

        # We have nothing to send
        if nb_ft_map_sent == 0:
            entropy_coded_byte = b''
        # We have something to send
        else:
            # Compute the CDF for all symbols
            output_cdf = self.compute_cdf({
                'mode': mode,
                'sigma': sigma_non_zero,
                'data_dimension': x_non_zero.size(),
            })

            # CDF must be on CPU
            output_cdf = output_cdf.cpu()

            # Shift x from [-max_val, max_val - 1] to [0, 2 * max_val -1]
            symbol = (x_non_zero + self.AC_MAX_VAL).cpu().to(torch.int16)
            entropy_coded_byte = torchac.encode_float_cdf(output_cdf, symbol, check_input_bounds=True)
        
        # Add the entropy coded part to the bytestream
        byte_to_write += entropy_coded_byte

        # Append the number of bytes at the beginning
        byte_to_write = len(byte_to_write).to_bytes(4, byteorder='big') + byte_to_write

        # If we're writting codecnet_z and we don't have written mofnet before
        # we need to specify that we've not written anything for mofnet_z
        # and for mofnet_y
        if latent_name == 'codecnet_z' and not (os.path.isfile(bitstream_path)):
            # 0 byte for mofnet z
            byte_to_write = int(0).to_bytes(4, byteorder='big') + byte_to_write
            # 0 byte for codecnet_z
            byte_to_write = int(0).to_bytes(4, byteorder='big') + byte_to_write

        if os.path.isfile(bitstream_path):
            old_size = os.path.getsize(bitstream_path)
        else:
            old_size = 0
        with open(bitstream_path, 'ab') as fout:
            fout.write(byte_to_write)
        new_size = os.path.getsize(bitstream_path)
        
        # Check wether we have the same rate as in real life
        if flag_debug:
            if mode == 'laplace':
                # Compute the theoretical rate
                b = sigma / torch.sqrt(torch.tensor([2.0], device=x.device))
                my_pdf = Laplace(torch.zeros_like(b, device=b.device), b)
                proba = torch.clamp(my_pdf.cdf(x + 0.5) - my_pdf.cdf(x - 0.5), 2 ** -16, 1.)
            elif mode == 'pmf':
                proba = torch.clamp(self.balle_pdf_estim(x), 2 ** -16, 1.)
                
            estimated_rate = (-torch.log2(proba).sum() / (8000)).item() + 1e-3 # Avoid having a perfect zero rate: minimum one byte

            real_rate = len(byte_to_write) / 1000
            rate_overhead = (real_rate / estimated_rate - 1) * 100
            absolute_overhead = real_rate - estimated_rate

            print('Arithmetic coding of      : ' + str(bitstream_path.split('/')[-1].rstrip(BITSTREAM_SUFFIX)) + ' ' + latent_name)
            print('Number of ft. maps sent   : ' + str(nb_ft_map_sent))            
            print('Bitrate estimation [kByte]: ' + '%.3f' % (estimated_rate))
            print('Real bitstream     [kByte]: ' + '%.3f' % (real_rate))
            print('Rate overhead          [%]: ' + '%.1f' % (rate_overhead))
            print('Absolute overhead  [Kbyte]: ' + '%.3f' % (absolute_overhead))
            print('Header overhead     [byte]: ' + '%.1f' % (header_overhead))
            print('Nb. bytes in file   [byte]: ' + '%.1f' % (new_size - old_size))


        # Check that entropy coding is lossless
        if flag_debug:
            x_decoded = self.decode({
                'mode': mode,
                'sigma': sigma,
                'bitstream_path': bitstream_path,
                'data_dim': x.size(),
                'device': x.device,
                'flag_debug': flag_debug,
                'latent_name': latent_name,
                'flag_md5sum': flag_md5sum,
            })

            if torch.all(torch.eq(x, x_decoded)):
                print('Ok! Entropy coding is lossless\n')
            else:
                print('-' * 80)
                print('Ko! Entropy coding is not lossless: ' + str((x_decoded - x).abs().sum()) + '\n')
                print('-' * 80)

    def decode(self, param):
        
        DEFAULT_PARAM = {
            # Mode is either <laplace> or <pmf>
            'mode': 'laplace',
            # If mode == 'laplace', we need the parameters mu and sigma
            'sigma': None,
            # Name (absolute path) of the bitstream
            'bitstream_path': None,
            # Dimension of the data to decode, as a tuple (B, C, H, W)
            'data_dim': None,
            # On which device the code will run
            'device': 'cpu',
            # Debug. If true, print specific stuff
            'flag_debug': True,
            # Specify what quantity we're entropy coding.
            # It can be <mofnet_y>, <mofnet_z>, <codecnet_y>, codecnet_z>.
            # This is needed for the decoding part, to know which part of the
            # bitstream we should read.
            # Bitstream structure for each frame is detailed above.
            'latent_name': '',
            # A feature-wise md5sum is included in the bitstream, verify
            # that the decoded version is identical to the provided md5sum.
            'flag_md5sum': False,
        }

        mode = get_value('mode', param, DEFAULT_PARAM)
        sigma = get_value('sigma', param, DEFAULT_PARAM)
        bitstream_path = get_value('bitstream_path', param, DEFAULT_PARAM)
        data_dim = get_value('data_dim', param, DEFAULT_PARAM)
        device = get_value('device', param, DEFAULT_PARAM)
        flag_debug = get_value('flag_debug', param, DEFAULT_PARAM)
        latent_name = get_value('latent_name', param, DEFAULT_PARAM)
        flag_md5sum = get_value('flag_md5sum', param, DEFAULT_PARAM)

        if not(bitstream_path.endswith(BITSTREAM_SUFFIX)):
            bitstream_path += BITSTREAM_SUFFIX

        # Read byte-stream
        with open(bitstream_path, 'rb') as fin:
            byte_stream = fin.read()

        # Check how many bytes we have to skip
        if latent_name == 'mofnet_z':
            nb_byte_skip = 0
        else:
            nb_byte_skip = int.from_bytes(byte_stream[0:4], byteorder='big')
            byte_stream = byte_stream[nb_byte_skip + 4:]

            # Stop here if we're decoding mofnet_y else, continue
            if latent_name in ['codecnet_y', 'codecnet_z']:
                nb_byte_skip = int.from_bytes(byte_stream[0:4], byteorder='big')
                byte_stream = byte_stream[nb_byte_skip + 4:]

                if latent_name == 'codecnet_y':
                    nb_byte_skip = int.from_bytes(byte_stream[0:4], byteorder='big')
                    byte_stream = byte_stream[nb_byte_skip + 4:]

        # Once we've skipped the appropriate number of bytes, the first 4 bytes gives us the number of
        # bytes we have to read for decoding this latent
        nb_byte_to_read = int.from_bytes(byte_stream[0:4], byteorder='big')
        byte_stream = byte_stream[4:]

        # We can now cut the part of the bytestream we want to conserve
        byte_stream = byte_stream[:nb_byte_to_read]

        # Check the encoder feature-wise md5sum
        if flag_md5sum:
            encoder_md5sum = byte_stream[0:32].decode().encode()
            byte_stream = byte_stream[32:]

        # We haven't transmitted the non-zero feature maps
        # Retrieve the non-zero indices, cut the bytestream after
        # each reading
        if mode == 'laplace':
            nb_ft_map_sent = byte_stream[0]
            byte_stream = byte_stream[1:]
            
            if nb_ft_map_sent != 0:
                # Retrieve the indices of the feature maps transmitted
                idx_non_zero = []
                for i in range(nb_ft_map_sent):
                    idx_non_zero.append(byte_stream[0])
                    byte_stream = byte_stream[1:]

                # Keep mu and sigma only for the non zero ft map.
                # mu_non_zero = mu[:, idx_non_zero, :, :]
                sigma_non_zero = sigma[:, idx_non_zero, :, :]
                nb_ft_map_sent = len(idx_non_zero)
        
                # Compute the CDF for all symbols
                output_cdf = self.compute_cdf({
                    'mode': mode,
                    'sigma': sigma_non_zero,
                    'data_dimension': data_dim,
                })
                # print('Decoder device CDF: ' + str(output_cdf.abs().sum()))
                # cdf must be on cpu
                output_cdf = output_cdf.cpu()
                # print('Decoder CPU CDF   : ' + str(output_cdf.abs().sum()))

                # Decode byte stream
                symbol = torchac.decode_float_cdf(
                    output_cdf, byte_stream, needs_normalization=True
                )

                # Shift back symbol
                x_decoded_non_zero = (symbol - self.AC_MAX_VAL)

                # Create empty x and add the non-zero data to it
                x_decoded = torch.zeros_like(sigma, device='cpu').to(torch.int16)
                x_decoded[:, idx_non_zero, :, :] = x_decoded_non_zero
                x_decoded = x_decoded.to(torch.float).to(device)
            else:
                x_decoded = torch.zeros_like(sigma, device=device)

        elif mode == 'pmf':
            # Compute the CDF for all symbols
            output_cdf = self.compute_cdf({
                'mode': mode,
                'sigma': sigma,
                'data_dimension': data_dim,
            })

            # print('Decoder device CDF: ' + str(output_cdf.abs().sum()))
            # cdf must be on cpu
            output_cdf = output_cdf.cpu()
            # print('Decoder CPU CDF   : ' + str(output_cdf.abs().sum()))

            # Decode byte stream
            symbol = torchac.decode_float_cdf(output_cdf, byte_stream)

            # Shift back symbol
            x_decoded = (symbol - self.AC_MAX_VAL).to(torch.float).to(device)
        
        # # Check the decoder feature-wise md5sum
        if flag_md5sum:
            x_decoder = x_decoded.to('cpu').to(torch.int16).numpy().astype(int)
            np.savetxt( './tmp_tensor.npy', x_decoder.flatten())
            dec_md5 = compute_md5sum({'in_file': './tmp_tensor.npy'}).encode()
            os.system('rm ./tmp_tensor.npy')

            if dec_md5 != encoder_md5sum:
                print('[Error] lossy arithmetic coding for')
                print('\t' + bitstream_path + ' ' + latent_name)
                print('-' * 80)
            else:
                print('All good for ' + bitstream_path + ' ' + latent_name)

        return x_decoded
