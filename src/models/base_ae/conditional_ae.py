# Software Name: AIVC
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

# Built-in modules
import os

# Third party modules
import torch
from torch.nn import Module

# Custom modules
from func_util.console_display import print_log_msg
from func_util.nn_util import get_value
from func_util.GOP_structure import FRAME_I, FRAME_P, FRAME_B
from layers.misc.misc_layers import Quantizer, PdfParamParameterizer
from layers.entropy_coding.pdf_estimator import BallePdfEstim,\
                                                ParametricPdf
from layers.entropy_coding.entropy_coder import EntropyCoder
from layers.multi_rate.gain_matrix import GainMatrix
from models.base_ae.ae_building_blocks import BaseEncoder, BaseDecoder,\
                                       BaseEncoderHP, BaseDecoderHP,\
                                       ChengEncoder, ChengDecoder,\
                                       ChengEncoderHp, ChengDecoderHp

class ConditionalNet(Module):

    def __init__(self, model_param):
        super(ConditionalNet, self).__init__()

    def __init__(self, model_param):

        super(ConditionalNet, self).__init__()
        # ===== RETRIEVE PARAMETERS ===== #
        self.model_param = model_param

        self.model_type = self.model_param.get('model_type')
        self.attention_type = self.model_param.get('attention_type')
        self.nb_ft_y = self.model_param.get('nb_ft_y')
        self.nb_ft_z = self.model_param.get('nb_ft_z')

        # Computed beforehand
        self.in_c = self.model_param.get('in_c')
        self.out_c = self.model_param.get('out_c')
        self.in_c_shortcut_y = self.model_param.get('in_c_shortcut_y')
        self.out_c_shortcut_y = self.model_param.get('out_c_shortcut_y')
        self.out_c_shortcut_z = self.model_param.get('out_c_shortcut_z')

        # Non-linearities
        self.nl_main_enc = self.model_param.get('nl_main_enc')
        self.nl_main_dec = self.model_param.get('nl_main_dec')
        self.nl_hp_enc = self.model_param.get('nl_hp_enc')
        self.nl_hp_dec = self.model_param.get('nl_hp_dec')

        # Biases for the last layers
        self.bias_last_main_enc = self.model_param.get('bias_last_main_enc')
        self.bias_last_hp_enc = self.model_param.get('bias_last_hp_enc')
        self.bias_last_main_shortcut =\
            self.model_param.get('bias_last_main_shortcut')
        self.bias_last_hp_shortcut =\
            self.model_param.get('bias_last_hp_shortcut')

        # ==== GAIN P B PARAMETERS ===== #
        self.flag_gain_p_b = self.model_param.get('flag_gain_p_b')
        self.nb_rate_point = self.model_param.get('nb_rate_point')
        # ==== GAIN P B PARAMETERS ===== #
        
        
        self.flag_multi_rate = self.model_param.get('flag_multi_rate')
        self.nb_ft_conv = self.model_param.get('nb_ft_conv')
        self.k_size = self.model_param.get('k_size')
        self.ec_mode = self.model_param.get('ec_mode')
        self.upscaling_mode = self.model_param.get('upscaling_mode')
        self.upscaling_mode_hp = self.model_param.get('upscaling_mode_hp')
        # ===== RETRIEVE PARAMETERS ===== #

        # ===== DEPRECATED FLAGS ===== #
        # self.y_arm_type = self.model_param.get('y_arm_type')
        self.z_arm_type = self.model_param.get('z_arm_type')
        # self.flag_ar_y = False if self.y_arm_type == 'no_ar' else True
        self.flag_ar_z = False if self.z_arm_type == 'no_ar' else True
        # ===== DEPRECATED FLAGS ===== #

        # ===== COMPUTE VARIOUS FLAGS ===== #
        self.flag_binary_model = 'bin' in self.ec_mode.split('_')
        self.flag_shortcut_y = (self.out_c_shortcut_y != 0)
        self.flag_shortcut_z = (self.out_c_shortcut_z != 0)

        # How many parameters per component of the mixture
        if 'gamma' in self.ec_mode.split('_'):
            # mu sigma gamma weight
            nb_param_per_mixture_component = 4
        elif self.flag_binary_model:
            nb_param_per_mixture_component = 4
        else:
            # mu sigma weight
            nb_param_per_mixture_component = 3

        if 'two' in self.ec_mode.split('_'):
            K = 2
        elif 'three' in self.ec_mode.split('_'):
            K = 3
        else:
            K = 1

        # ! K is alway set to one when working with binary model
        if self.flag_binary_model:
            nb_param_pdf_per_latent = nb_param_per_mixture_component
        else:
            # We have this (v) number of parameters for each latent y_i distrib
            # substraction by 1 because we need (K - 1) weight for the mixture
            # as their sum is always 1.
            nb_param_pdf_per_latent = K * nb_param_per_mixture_component - 1

        self.nb_output_hp = self.nb_ft_y * nb_param_pdf_per_latent

        if self.model_type == 'base_add':
            self.nb_ft_in_dec = self.nb_ft_y
        else:
            self.nb_ft_in_dec = self.nb_ft_y + self.out_c_shortcut_y
        self.nb_ft_in_dec_hp = self.nb_ft_z + self.out_c_shortcut_z
        # ===== COMPUTE VARIOUS FLAGS ===== #

        # ===== NETWORK ARCHITECTURE ===== #
        param_main_enc = {
            'in_c': self.in_c,
            'out_c': self.nb_ft_y,
            'nb_ft': self.nb_ft_conv,
            'attention': self.attention_type,
            'flag_bias_last_layer': self.bias_last_main_enc,
        }

        param_main_dec = {
            'in_c': self.nb_ft_in_dec,
            'out_c': self.out_c,
            'nb_ft': self.nb_ft_conv,
            'attention': self.attention_type,
        }

        param_hp_enc = {
            'in_c': self.nb_ft_y,
            'out_c': self.nb_ft_z,
            'nb_ft': self.nb_ft_conv,
            'flag_bias_last_layer': self.bias_last_hp_enc,
        }

        param_hp_dec = {
            'in_c': self.nb_ft_in_dec_hp,
            'out_c': self.nb_output_hp,
            'nb_ft': self.nb_ft_conv,
        }

        param_shortcut_y = {
            'in_c': self.in_c_shortcut_y,
            'out_c': self.out_c_shortcut_y,
            'nb_ft': self.nb_ft_conv,
            'attention': self.attention_type,
            'flag_bias_last_layer': self.bias_last_main_shortcut,
        }

        param_shortcut_z = {
            'in_c': self.out_c_shortcut_y,
            'out_c': self.out_c_shortcut_z,
            'nb_ft': self.nb_ft_conv,
            'attention': self.attention_type,
            'flag_bias_last_layer': self.bias_last_hp_shortcut,
        }

        if self.model_type == 'cheng':
            main_encoder = ChengEncoder
            main_decoder = ChengDecoder
            hp_encoder = ChengEncoderHp
            hp_decoder = ChengDecoderHp
        elif 'base' in self.model_type.split('_'):
            main_encoder = BaseEncoder
            main_decoder = BaseDecoder
            hp_encoder = BaseEncoderHP
            hp_decoder = BaseDecoderHP

            # ==== ADD BASE SPECIFIC PARAMETERS HERE ==== #
            param_main_enc['nl_type'] = self.nl_main_enc
            param_main_dec['nl_type'] = self.nl_main_dec
            param_shortcut_y['nl_type'] = self.nl_main_enc

            param_hp_enc['nl_type'] = self.nl_hp_enc
            param_hp_dec['nl_type'] = self.nl_hp_dec
            param_shortcut_z['nl_type'] = self.nl_hp_enc
            # ==== ADD BASE SPECIFIC PARAMETERS HERE ==== #
          
        self.g_a = main_encoder(param_main_enc)

        self.g_s = main_decoder(param_main_dec)

        self.h_a = hp_encoder(param_hp_enc)

        self.h_s = hp_decoder(param_hp_dec)

        # If shortcut transforms are needed
        if self.flag_shortcut_y or self.flag_shortcut_z:
            self.g_a_ref = main_encoder(param_shortcut_y)
            if self.flag_shortcut_z:
                self.h_a_ref = hp_encoder(param_shortcut_z)

        # Residual implementation, needs two decoder: 
        # One for y_enc => delta_v, alpha and beta
        # One for y_shortcut => interpol_v
        # With v = interpol_v and delta_v
        if self.model_type == 'base_add':
            param_shortcut_dec = {
                'in_c': self.out_c_shortcut_y,
                'out_c': 4,    # 2 optical flows = 2 * 2
                'nb_ft': self.nb_ft_conv,
                'attention': self.attention_type,
                'flag_bias_last_layer': True,
            }
            self.shortcut_dec = main_decoder(param_shortcut_dec)

        self.pdf_parameterizer = PdfParamParameterizer(self.ec_mode, self.nb_ft_y)
        self.quantizer = Quantizer()

        # Needed at least for z
        self.entropy_coder = EntropyCoder()

        # Use a PMF + mu for y
        if self.ec_mode == 'mupmf':
            self.pdf_y = BallePdfEstim(self.nb_ft_y, 'pmf')
        else:
            self.pdf_y = ParametricPdf(self.ec_mode)

        # For z
        self.pdf_z = BallePdfEstim(self.nb_ft_z, 'pmf')

        # ================== GAIN VECTORS FOR MULTI-RATE =================== #
        self.gain_I = GainMatrix({
            'N': self.nb_rate_point,
            'nb_ft': self.nb_ft_y,
            'initialize_to_one': True,
            'scalar_gain': False,
        })

        # Gain vector for P and B if needed
        if self.flag_gain_p_b:
            self.gain_P = GainMatrix({
                'N': self.nb_rate_point,
                'nb_ft': self.nb_ft_y,
                'initialize_to_one': True,
                'scalar_gain': False,
            })

            self.gain_B = GainMatrix({
                'N': self.nb_rate_point,
                'nb_ft': self.nb_ft_y,
                'initialize_to_one': True,
                'scalar_gain': False,
            })
        # ================== GAIN VECTORS FOR MULTI-RATE =================== #


    def forward(self, param):
        DEFAULT_PARAM = {
            # Input of the encoder (4 dimensional)
            'in_enc': None,
            # Input of the shortcut (4 dimensional)
            'in_shortcut': None,
            # For multi-rate, not used for now
            'idx_rate': 0.,
            # A boolean indicating wether we use the shortcut
            # transform for each of the B examples of the batch
            'use_shortcut_vector': None,
            # A scalar indicating the type of the frame for
            # each of the B examples which are either: FRAME_I, FRAME_P or 
            # FRAME_B
            'frame_type': None,
            # If true, we generate a bitstream at the end
            'generate_bitstream': False,
            # Path where the bistream is written
            'bitstream_path': '',
            # Set to true to generate more stuff, useful for debug
            'flag_bitstream_debug': False,
            # Specify what quantity we're entropy coding.
            # It can be <mofnet>, <codecnet>.
            # This is needed for the decoding part, to know which part of the
            # bitstream we should read.
            # Bitstream structure for each frame is detailed above.
            'latent_name': '',
        }

        in_enc = get_value('in_enc', param, DEFAULT_PARAM)
        raw_in_shortcut = get_value('in_shortcut', param, DEFAULT_PARAM)
        idx_rate = get_value('idx_rate', param, DEFAULT_PARAM)
        use_shortcut_vector = get_value('use_shortcut_vector', param, DEFAULT_PARAM)
        frame_type = get_value('frame_type', param, DEFAULT_PARAM)
        generate_bitstream = get_value('generate_bitstream', param, DEFAULT_PARAM)
        bitstream_path = get_value('bitstream_path', param, DEFAULT_PARAM)
        flag_bitstream_debug = get_value('flag_bitstream_debug', param, DEFAULT_PARAM)
        latent_name = get_value('latent_name', param, DEFAULT_PARAM)

        net_out = {}
        cur_device = in_enc.device

        # Go to the latent space
        y_code = self.g_a(in_enc)

        # ===== SHORTCUT TRANSFORM PRESENT ====== #
        if self.flag_shortcut_y:
            # We don't need the shortcut at all (for no frame)
            if not(use_shortcut_vector):
                B, _, H_y, W_y = y_code.size()
                y_shortcut = torch.zeros(B, self.out_c_shortcut_y, H_y, W_y, device=cur_device)
            # We need the shortcut for some frames (but not necessary all)
            else:
                y_shortcut = self.g_a_ref(raw_in_shortcut)
        # ===== SHORTCUT TRANSFORM PRESENT ====== #

        # ===== Y GAIN VECTOR PT. 1 ===== #
        # Select the appropriate gain matrix
        # We don't have a dedicated gain matrix for P or B frames
        gain_matrix_input = {
            'x': y_code, 'idx_rate': idx_rate, 'mode': 'enc'
        }
        
        if not(self.flag_gain_p_b):
            net_out_gain_1 = self.gain_I(gain_matrix_input)
        else:
            if frame_type == FRAME_I:
                net_out_gain_1 = self.gain_I(gain_matrix_input)
            elif frame_type == FRAME_P:
                net_out_gain_1 = self.gain_P(gain_matrix_input)
            elif frame_type == FRAME_B:
                net_out_gain_1 = self.gain_B(gain_matrix_input)
        
        y_code_scaled = net_out_gain_1.get('output')

        if generate_bitstream:
            y_code_scaled = torch.clamp(
                y_code_scaled, -self.ac.AC_MAX_VAL, self.ac.AC_MAX_VAL - 1
            )
        # ===== Y GAIN VECTOR PT. 1 ===== #


        # ===== HYPERPRIOR PART ===== #
        # Compute and transmit hyperprior
        z_scaled = self.h_a(y_code_scaled)

        # We clamp the non-quantized version of z, therefore:
        #   z is in [-max_val, max_val - 1]
        #   z_hat = round(z) is also in [-max_val, max_val - 1]
        if generate_bitstream:
            z_scaled = torch.clamp(z_scaled, -self.ac.AC_MAX_VAL, self.ac.AC_MAX_VAL - 1)            

        z_hat_scaled = self.quantizer(z_scaled)

        # Balle PDF Estim CDF
        p_z_hat = self.pdf_z(z_hat_scaled)
        rate_z_hat = self.entropy_coder(p_z_hat, z_hat_scaled)

        # Theoretically after sending
        z_hat = z_hat_scaled

        # ===== SHORTCUT TRANSFORM ABSENT ====== #
        if not(self.flag_shortcut_y):
            y_shortcut = torch.zeros_like(y_code_scaled, device=cur_device)
        # ===== SHORTCUT ABSENT ====== #

        # ===== DECODE Z WITH THE SHORTCUT TRANSFORM ===== #
        z_to_decode = z_hat

        # Due to upscaling z_hat_decoded could be bigger than y_ref
        tgt_size2 = y_code.size()[2]
        tgt_size3 = y_code.size()[3]

        decoded_z = self.h_s(z_to_decode)[:, :, :tgt_size2, :tgt_size3]
        pdf_param_y = self.pdf_parameterizer(decoded_z)
        # ===== DECODE Z WITH THE SHORTCUT TRANSFORM ===== #

        centered_y = self.quantizer(y_code_scaled - pdf_param_y[0].get('mu'))
        
        if generate_bitstream:
            centered_y = torch.clamp(centered_y, -self.ac.AC_MAX_VAL, self.ac.AC_MAX_VAL - 1)
        
        p_y_hat = self.pdf_y(centered_y, pdf_param_y, zero_mu=True)
        rate_y_hat = self.entropy_coder(p_y_hat, centered_y)

        # Bring back mu
        y_decoder = centered_y + pdf_param_y[0].get('mu')
        
        # Theoretically after sending
        # ===== Y GAIN VECTOR PT. 2 ===== #
        gain_matrix_input_2 = {
            'x': y_decoder, 'idx_rate': idx_rate, 'mode': 'dec'
        }        

        # Get the correct gain matrix        
        if not(self.flag_gain_p_b):
            net_out_gain_2 = self.gain_I(gain_matrix_input_2)
        else:
            if frame_type == FRAME_I:
                net_out_gain_2 = self.gain_I(gain_matrix_input_2)
            elif frame_type == FRAME_P:
                net_out_gain_2 = self.gain_P(gain_matrix_input_2)
            elif frame_type == FRAME_B:
                net_out_gain_2 = self.gain_B(gain_matrix_input_2)
        
        y_code_hat = net_out_gain_2.get('output')
        # ===== Y GAIN VECTOR PT. 2 ===== #

        # * ==== Decode y with the shortcut connexion if needed
        if self.flag_shortcut_y:
            y_to_decode = torch.cat((y_code_hat, y_shortcut), dim=1)
        else:
            y_to_decode = y_code_hat

        decoded_y = self.g_s(y_to_decode)
        x_hat = decoded_y

        # ! x_hat is the main output. The different feature maps of x_hat can be splitted
        # ! later if needed.
        net_out['x_hat'] = x_hat
        net_out['rate_y'] = rate_y_hat
        net_out['rate_z'] = rate_z_hat

        if generate_bitstream:
            if flag_bitstream_debug:
                print('Encoder idx_rate: ' + str(idx_rate))

            # Encode z
            self.ac.encode({
                'x': z_hat_scaled,
                'mode': 'pmf',
                'bitstream_path': bitstream_path,
                'flag_debug': flag_bitstream_debug,
                'latent_name': latent_name + '_z'
            })
            # Encode y
            self.ac.encode({
                'mode': 'laplace',
                'x': centered_y,
                'sigma': pdf_param_y[0].get('sigma'),
                'bitstream_path': bitstream_path,
                'flag_debug': flag_bitstream_debug,
                'latent_name': latent_name + '_y',
            })
            
        return net_out


    def print_debug_dic(self, debug_dic):
        for k, v in debug_dic.items():
            print('\n' + k + ':')
            if isinstance(v, dict):
                for cur_k, cur_v in v.items():
                    print('\n\t' + cur_k + ':')
                    print('\t\tMax = ' + str(cur_v.abs().max()))
                    print('\t\tMin = ' + str(cur_v.abs().min()))
                    print('\t\tAverage = ' + str(cur_v.abs().mean()))
            elif isinstance(v, list):
                for cur_k, cur_v in zip(['mu', 'sigma'], v):
                    print('\n\t' + cur_k + ':')
                    print('\t\tMax = ' + str(cur_v.abs().max()))
                    print('\t\tMin = ' + str(cur_v.abs().min()))
                    print('\t\tAverage = ' + str(cur_v.abs().mean()))

            else:
                print('\tMax = ' + str(v.abs().max()))
                print('\tMin = ' + str(v.abs().min()))
                print('\tAverage = ' + str(v.abs().mean()))

    def print_debug_param(self):
        for name, param in self.named_parameters():
            print('\n' + name)
            print('\tMax = ' + str(param.abs().max()))
            print('\tMin = ' + str(param.abs().min()))
            print('\tAverage = ' + str(param.abs().mean()))
