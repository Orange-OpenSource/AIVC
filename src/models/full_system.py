# Software Name: AIVC
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

import os
import torch
from torch.nn import Module, ReplicationPad2d

# Custom modules
from func_util.console_display import print_log_msg, print_dic_content
from func_util.img_processing import get_y_u_v, save_yuv_separately, cast_before_png_saving
from func_util.nn_util import get_value, dic_zeros_like, push_dic_to_device
from func_util.cluster_mngt import COMPUTE_PARAM
from func_util.GOP_structure import get_name_frame_code, get_depth_gop,\
                                    FRAME_B, FRAME_P, FRAME_I
from models.codec_net import CodecNet
from models.mode_net import ModeNet
from models.motion_compensation import MotionCompensation
from layers.ae.ae_layers import InputLayer, OutputLayer

from real_life.utils import GOP_HEADER_SUFFIX
from real_life.check_md5sum import write_md5sum
from real_life.header import write_gop_header
from real_life.cat_binary_files import cat_one_gop


class FullNet(Module):

    def __init__(self, model_param):
        super(FullNet, self).__init__()
        # ===== RETRIEVE PARAMETERS ===== #
        self.model_param = model_param
        self.name = self.model_param.get('net_name')
        self.current_lr = self.model_param.get('initial_lr')

        # Retrieve distortion metrics (<mse>, <l1> or <ms_ssim>)
        self.dist_loss = self.model_param.get('dist_loss')
        self.warping_mode = self.model_param.get('warping_mode')
        # ===== COMPUTE VARIOUS FLAGS ===== #
        # By default, all flags are False

        # Construct alternate training list
        self.training_mode_list = ['mode_net', 'codec_net']
        # ===== COMPUTE VARIOUS FLAGS ===== #

        # ===== OTHERS PARAMETERS ===== #
        # Prefix used for saving during training
        self.train_save_idx = 0
        # Used to save the number of epochs done
        self.nb_epoch_done = 0
        self.batch_cnt = 0
        # ===== OTHERS PARAMETERS ===== #

        # ===== SUB NETWORKS ===== #
        self.in_layer = InputLayer()
        self.out_layer = OutputLayer()

        # Hard-wire some quantities here
        # For CodecNet, if there is a shortcut transform for y and/or z, we
        # need to have 3 more features at the input (in_c_shortcut_y). This
        # results in 6 total features for input (in_c).
        if model_param.get('codec_net_param').get('out_c_shortcut_y') or\
           model_param.get('codec_net_param').get('out_c_shortcut_z'):
            model_param.get('codec_net_param')['in_c_shortcut_y'] = 3
            model_param.get('codec_net_param')['in_c'] = 6

        # Same things for mode_net
        if model_param.get('mode_net_param').get('out_c_shortcut_y') or\
           model_param.get('mode_net_param').get('out_c_shortcut_z'):
            model_param.get('mode_net_param')['in_c_shortcut_y'] = 3

        model_param.get('mode_net_param')['in_c'] = 6
        # Always 3, because we output an image
        model_param.get('codec_net_param')['out_c'] = 3
        # 3 = alpha + v_x + v_y
        model_param.get('mode_net_param')['out_c'] = 3

        self.codec_net = CodecNet(model_param.get('codec_net_param'))
        self.mode_net = ModeNet(model_param.get('mode_net_param'))
        self.motion_compensation = MotionCompensation(model_param.get('motion_comp_param'))
        # ===== SUB NETWORKS ===== #

        print_log_msg(
            'INFO', '__init__ FullNet', 'network_name', self.name
        )
        print_log_msg(
            'DEBUG', '__init__ FullNet', 'training', self.training
        )
        print_log_msg(
            'DEBUG', '__init__ FullNet', 'warping_mode', self.warping_mode
        )
        print_log_msg('DEBUG', '__init__ FullNet', 'state', 'done')

        # ===== PRINT MISC STUFF ===== #
        print_dic_content(self.model_param, dic_name='Codec Model parameters')
        print_log_msg('INFO', '__init__ Codec', 'Printing entire', ' network')
        if not(COMPUTE_PARAM.get('flag_quiet')):
            print(self)
        # After architecture declaration, print number of parameters
        print_dic_content(self.get_nb_param(), dic_name='Codec Nb. parameters')
        # ===== PRINT MISC STUFF ===== #

    def get_nb_param(self):
        nb_param_dic = {}

        accumulator = 0
        for name, param in self.named_parameters():
            nb_param_dic[name] = param.numel()
            accumulator += param.numel()

        nb_param_dic['total'] = accumulator

        return nb_param_dic

    def forward(self, param):
        """
        Compress one frame.
        """
        DEFAULT_PARAM = {
            # YUV dictionnary to encode/decode
            'code_dic': None,
            # YUV dictionnary of the previous frame
            'prev_dic': None,
            # YUV dictionnary of the next frame
            'next_dic': None,
            # A tensor of dimension B, indicating the type of the frame for
            # each of the B examples which are either: FRAME_I, FRAME_P or
            # FRAME_B
            'frame_type': None,
            # If not None: override the ModeNet y with external y
            'external_y_modenet': None,
            # If not None: override the CodecNet y with external y
            'external_y_codecnet': None,
            # For multi-rate
            'idx_rate': 0.,
            # If True, alpha is equal to 1 for the entire frame (everything goes
            # into the CodecNet and ignore skip mode)
            # ! Deprecated
            'flag_no_copy': False,
            # If True, alpha is equal to 0 for the entire frame (everything goes
            # into the Skip Mode and ignore CodecNet)
            # ! Deprecated
            'flag_no_coder': False,
            'generate_bitstream': False,
            # Path where the bistream is written
            'bitstream_path': '',
            # Set to true to generate more stuff, useful for debug
            'flag_bitstream_debug': False,
        }

        net_out = {}

        # ===== RETRIEVE INPUTS ===== #
        p = get_value('prev_dic', param, DEFAULT_PARAM)
        c = get_value('code_dic', param, DEFAULT_PARAM)
        n = get_value('next_dic', param, DEFAULT_PARAM)
        frame_type = get_value('frame_type', param, DEFAULT_PARAM)
        external_y_modenet = get_value('external_y_modenet', param, DEFAULT_PARAM)
        external_y_codecnet = get_value('external_y_codecnet', param, DEFAULT_PARAM)
        idx_rate = get_value('idx_rate', param, DEFAULT_PARAM)
        generate_bitstream = get_value('generate_bitstream', param, DEFAULT_PARAM)
        bitstream_path = get_value('bitstream_path', param, DEFAULT_PARAM)
        flag_bitstream_debug = get_value('flag_bitstream_debug', param, DEFAULT_PARAM)
        # ===== RETRIEVE INPUTS ===== #

        # ===== PRE-PROCESSING ===== #
        prev_ref = self.in_layer(p)
        next_ref = self.in_layer(n)
        code = self.in_layer(c)
        # ===== PRE-PROCESSING ===== #

        B, C, H, W = prev_ref.size()
        cur_device = prev_ref.device

        # ===== MODE NET ===== #
        mode_net_input = {
            'code': code,
            'prev': prev_ref,
            'next': next_ref,
            'external_y': external_y_modenet,
            'idx_rate': idx_rate,
            'frame_type': frame_type,
            'use_shortcut_vector': torch.ones(B, device=cur_device).bool(),
            'generate_bitstream': generate_bitstream,
            'bitstream_path': bitstream_path,
            'flag_bitstream_debug': flag_bitstream_debug,
        }

        # We always ignore useless modules that is:
        #   - I-frame: MOFNet and CodecNet shortcut
        #   - P-frame: MOFNet shortcut
        #   - B-frame: Nothing

        # I-frame: skip the entire MOFNet
        if frame_type == FRAME_I:
            # No rate because we didn't use MOFNet
            # Dummy net_out_mode tensor
            net_out_mode = {}
            net_out_mode['rate_y'] = torch.zeros(1, 1, 1, 1, device=cur_device)
            net_out_mode['rate_z'] = torch.zeros(1, 1, 1, 1, device=cur_device)
            net_out_mode['alpha'] = torch.ones_like(code, device=cur_device)
            net_out_mode['beta'] = torch.ones_like(code, device=cur_device)
            net_out_mode['v_prev'] = torch.zeros(B, 2, H, W, device=cur_device)
            net_out_mode['v_next'] = torch.zeros(B, 2, H, W, device=cur_device)

        else:
            mode_net_input = {
                'code': code,
                'prev': prev_ref,
                'next': next_ref,
                'external_y': external_y_modenet,
                'idx_rate': idx_rate,
                'frame_type': frame_type,
                'generate_bitstream': generate_bitstream,
                'bitstream_path': bitstream_path,
                'flag_bitstream_debug': flag_bitstream_debug,
            }
            net_out_mode = self.mode_net(mode_net_input)

        # Retrieve value from net_out
        alpha = net_out_mode.get('alpha')
        beta = net_out_mode.get('beta')
        v_prev = net_out_mode.get('v_prev')
        v_next = net_out_mode.get('v_next')

        # alpha is not used for I frame
        if frame_type == FRAME_I:
            alpha[:, :, :, :] = 1.

        # Beta is only relevant for B frame
        if frame_type != FRAME_B:
            beta[:, :, :, :] = 1.
        # ===== MODE NET ===== #

        # ===== INTER PRED ===== #
        motion_comp_input = {
            'prev': prev_ref,
            'next': next_ref,
            'v_prev': v_prev,
            'v_next': v_next,
            'beta': beta,
            'interpol_mode': self.warping_mode,
        }

        motion_comp_out = self.motion_compensation(motion_comp_input)
        warped_ref = motion_comp_out.get('x_warp')
        skip_part = warped_ref * (1 - alpha)
        # ===== INTER PRED ===== #

        # ===== CODEC NET ===== #
        in_codec_net = alpha * code
        in_prediction_codec_net = alpha * warped_ref

        codec_net_input = {
            'code': in_codec_net,
            'prediction': in_prediction_codec_net,
            'external_y': external_y_codecnet,
            'idx_rate': idx_rate,
            'use_shortcut_vector': frame_type != FRAME_I, # Shortcut in CodecNet is useless for I-frame
            'frame_type': frame_type,
            'generate_bitstream': generate_bitstream,
            'bitstream_path': bitstream_path,
            'flag_bitstream_debug': flag_bitstream_debug,
        }
        net_out_codec = self.codec_net(codec_net_input)
        codec_part = net_out_codec.get('x_hat')
        # ===== CODEC NET ===== #

        result = codec_part + skip_part

        # ===== DOWNSCALING AND 420 STUFF ===== #
        x_hat = self.out_layer(result)

        # Downscaled version of u and v can be smaller than
        # their true size by one pixel
        # Difference in size should be of 0 or 1 pixel
        x_hat_y, x_hat_u, x_hat_v = get_y_u_v(x_hat)
        code_y, code_u, code_v = get_y_u_v(c)

        nb_pad_row = abs(code_u.size()[2] - x_hat_u.size()[2])
        nb_pad_col = abs(code_u.size()[3] - x_hat_u.size()[3])
        my_padding = ReplicationPad2d((0, nb_pad_col, 0, nb_pad_row))

        # Remove supplementary pixels if needed
        x_hat = {
            'y': x_hat_y,
            'u': my_padding(x_hat_u),
            'v': my_padding(x_hat_v),
        }

        if generate_bitstream:
            x_hat = cast_before_png_saving({'x': x_hat, 'data_type': 'yuv_dic'})
        # ===== DOWNSCALING AND 420 STUFF ===== #

        net_out['x_hat'] = x_hat

        # We don't use this in the loss, it's only here for logging purpose.
        # However as no optimizer goes through its gradient and reset it,
        # it keeps accumulating its computational graph.
        # Using detach() avoid this issue
        net_out['alpha'] = alpha.detach()
        net_out['beta'] = beta.detach()
        net_out['warping'] = warped_ref.detach()
        net_out['code'] = code.detach()

        net_out['codec_rate_y'] = net_out_codec.get('rate_y')
        net_out['codec_rate_z'] = net_out_codec.get('rate_z')
        net_out['mode_rate_y'] = net_out_mode.get('rate_y')
        net_out['mode_rate_z'] = net_out_mode.get('rate_z')

        return net_out

    def GOP_forward(self, param):
        """
        Compress a GOP.
        """
        DEFAULT_PARAM = {
            # The GOP structure defined as in func_util/GOP_structure.py
            'GOP_struct': None,
            # The name of the GOP structure, needed to construct the GOP header
            'GOP_struct_name': None,
            # The uncompressed frames (i.e. the frames to code), defined as:
            #   frame_0: {'y': tensor, 'u': tensor, 'v': tensor}
            #   frame_1: {'y': tensor, 'u': tensor, 'v': tensor}
            'raw_frames': None,
            # For multi-rate, not used for now
            'idx_rate': 0.,
            # Index of the GOP in the video. Scalar in [0, N]
            'index_GOP_in_video': 0,
            # If true, we generate a bitstream at the end
            'generate_bitstream': False,
            # Path of the directory in which we output the bitstream
            'bitstream_dir': '',
            # Frame index in the video of the first frame (I) of the
            # GOP.
            'real_idx_first_frame': 0,
            # Set to true to generate more stuff, useful for debug
            'flag_bitstream_debug': False,
        }

        # ========== RETRIEVE INPUTS ========== #
        GOP_struct = get_value('GOP_struct', param, DEFAULT_PARAM)
        GOP_struct_name = get_value('GOP_struct_name', param, DEFAULT_PARAM)
        raw_frames = get_value('raw_frames', param, DEFAULT_PARAM)
        idx_rate = get_value('idx_rate', param, DEFAULT_PARAM)
        index_GOP_in_video = get_value('index_GOP_in_video', param, DEFAULT_PARAM)
        generate_bitstream = get_value('generate_bitstream', param, DEFAULT_PARAM)
        bitstream_dir = get_value('bitstream_dir', param, DEFAULT_PARAM)
        real_idx_first_frame = get_value('real_idx_first_frame', param, DEFAULT_PARAM)
        flag_bitstream_debug = get_value('flag_bitstream_debug', param, DEFAULT_PARAM)
        # ========== RETRIEVE INPUTS ========== #

        # Outputs, each dic are structured as:
        # net_out: {'frame_0': {all entries}, 'frame_1': all entries}...
        net_out = {}

        # Number of temporal layers in the GOP, i.e. number of forward pass
        # to be performed. Get depth gop return the biggest coding order.
        N = get_depth_gop(GOP_struct)

        # Loop on the temporal layer. We go until N + 1 because if we have
        # a depth of 2, we want to code the depth of 0, 1 & 2!
        # Inside a temporal layer, all frames are independent so we code
        # them in parallel.
        for i in range(N + 1):
            # For a gop4 I - B - B - B - P,
            # i = 0 --> name_frame_code = ['frame_0']
            # i = 1 --> name_frame_code = ['frame_4']
            # i = 2 --> name_frame_code = ['frame_2']
            # i = 3 --> name_frame_code = ['frame_1', 'frame_3']

            # Return a list of one single element, so we remove the list
            name_frame_code = get_name_frame_code(GOP_struct, i)[0]
            # YUV dictionnary of the frame to code
            code = raw_frames.get(name_frame_code)
            # Type of the frame to code
            type_frame = GOP_struct.get(name_frame_code).get('type')

            # Get the references. We have a future ref. Retrieve the compressed version!
            if type_frame == FRAME_B:
                next_ref = net_out.get(GOP_struct.get(name_frame_code).get('next_ref')).get('x_hat')
            else:
                next_ref = dic_zeros_like(code)

            # We have a previous frame. Retrieve the compressed version
            if type_frame != FRAME_I:
                prev_ref = net_out.get(GOP_struct.get(name_frame_code).get('prev_ref')).get('x_hat')
            else:
                prev_ref = dic_zeros_like(code)

            # Push frame to device before the forward pass
            my_device = COMPUTE_PARAM.get('device')
            code = push_dic_to_device(code, my_device)
            prev_ref = push_dic_to_device(prev_ref, my_device)
            next_ref = push_dic_to_device(next_ref, my_device)

            model_input = {
                'code_dic': code,
                'prev_dic': prev_ref,
                'next_dic': next_ref,
                'idx_rate': idx_rate,
                'frame_type': type_frame,
                'generate_bitstream': generate_bitstream,
                # Complete bitstream path is: bitstream_dir/<idx_frame>
                # where idx_frame = real_idx_first_frame + X (X is found in frame_X)
                'bitstream_path': bitstream_dir + str(real_idx_first_frame + int(name_frame_code.split('_')[-1])),
                'flag_bitstream_debug': flag_bitstream_debug,
            }

            cur_net_out = self.forward(model_input)

            # Push frame to cpu after the forward pass
            my_device = 'cpu'
            code = push_dic_to_device(code, my_device)
            prev_ref = push_dic_to_device(prev_ref, my_device)
            next_ref = push_dic_to_device(next_ref, my_device)
            cur_net_out = push_dic_to_device(cur_net_out, my_device)
            torch.cuda.empty_cache()

            # Add the current frame dictionaries to the global ones
            net_out[name_frame_code] = cur_net_out

            # If we're generating a bitstream and if we're debugging this
            # bitstream, save the reconstructed PNGs (YUV) for the frame,
            # and compute the md5sum for the three PNGs.
            if generate_bitstream and flag_bitstream_debug:
                # /RealLife/debug/SequenceName/
                root_debug_path = '.' + '/'.join(bitstream_dir.split('/')[:-3]) + '/debug/' + bitstream_dir.split('/')[-2] + '/'

                # Real index of the frame in the video
                idx_frame = real_idx_first_frame + int(name_frame_code.split('_')[-1])

                # Save yuv as PNGs
                decoded_frame = cast_before_png_saving({
                    'x': net_out.get(name_frame_code).get('x_hat'), 'data_type': 'yuv_dic'
                })
                # print('Encoder decoded frame y: ' + str(decoded_frame.get('y').abs().sum()))
                save_yuv_separately(decoded_frame, root_debug_path + str(idx_frame))

                # Write the md5 in files and delete the PNGs
                for channel in ['y', 'u', 'v']:
                    write_md5sum({
                        'in_file': root_debug_path + str(idx_frame) + '_' + channel + '.png',
                        'out_file': root_debug_path + str(idx_frame) + '_' + channel + '.md5',
                    })
                    os.system('rm ' + root_debug_path + str(idx_frame) + '_' + channel + '.png')

        if generate_bitstream:
            # Write a header for this GOP
            # We need the GOP struct, and the x, y and z shape (we'll retrieve
            # this from the rate estimation tensor).

            write_gop_header({
                'header_path': bitstream_dir + str(index_GOP_in_video) + GOP_HEADER_SUFFIX,
                # Write only the last two dimensions (H and W)
                'data_dim': {
                    'x': raw_frames.get('frame_0').get('y').size()[2:],
                    'y': net_out.get('frame_0').get('codec_rate_y').size()[2:],
                    'z': net_out.get('frame_0').get('codec_rate_z').size()[2:],
                },
                'GOP_struct_name': GOP_struct_name,
                'idx_rate': idx_rate,
            })

            cat_one_gop({
                'bitstream_dir': bitstream_dir,
                'idx_gop': index_GOP_in_video,
            })

        return net_out
