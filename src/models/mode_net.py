# Built-in modules
import argparse

# Third party modules
import torch
from torch.nn import Module

# Custom modules
from func_util.console_display import print_log_msg
from func_util.nn_util import get_value
from func_util.GOP_structure import FRAME_B, FRAME_P, FRAME_I
from models.base_ae.conditional_ae import ConditionalNet


class ModeNet(Module):

    def __init__(self, model_param):

        super(ModeNet, self).__init__()

        # MofNet main encoder has 9 inputs (3 frames).
        # Mofnet encoder shortcut has 6 inputs (2 frames)
        # Mofnet main decoder has 6 outputs (2 optical flows + 2 pixel-wise weightings)
        model_param['in_c'] = 9
        model_param['out_c'] = 6
        model_param['in_c_shortcut_y'] = 6

        self.model_param = model_param
        self.mode_net = ConditionalNet(model_param)

        # ===== LOG MESSAGES ===== #
        print_log_msg('DEBUG', '__init__ ModeNet', 'state', 'done')
        # ===== LOG MESSAGES ===== #

    def forward(self, param):
        DEFAULT_PARAM = {
            # Tensor to encode/decode
            'code': None,
            # Previous reference frame x_{t-1}
            'prev': None,
            # Next reference x_{t+1}
            'next': None,
            # If True: return visu, a dictionnary full of visualisations
            'flag_visu': False,
            # If not None: override the y with external y
            'external_y': None,
            # For multi-rate, not used for now
            'idx_rate': 0.,
            # If true, we generate a bitstream at the end (and we don't go 
            # in the visu part?)
            'generate_bitstream': False,
            # Path where the bistream is written
            'bitstream_path': '',
            # A scalar, indicating the type of the frame for
            # each of the B examples which are either: FRAME_I, FRAME_P or 
            # FRAME_B
            'frame_type': None,
            # Set to true to generate more stuff, useful for debug
            'flag_bitstream_debug': False,
        }

        # Retrieve mode_net input data
        code = get_value('code', param, DEFAULT_PARAM)
        prev_ref = get_value('prev', param, DEFAULT_PARAM)
        next_ref = get_value('next', param, DEFAULT_PARAM)
        flag_visu = get_value('flag_visu', param, DEFAULT_PARAM)
        external_y = get_value('external_y', param, DEFAULT_PARAM)
        idx_rate = get_value('idx_rate', param, DEFAULT_PARAM)
        generate_bitstream = get_value('generate_bitstream', param, DEFAULT_PARAM)
        bitstream_path = get_value('bitstream_path', param, DEFAULT_PARAM)
        frame_type = get_value('frame_type', param, DEFAULT_PARAM)
        flag_bitstream_debug = get_value('flag_bitstream_debug', param, DEFAULT_PARAM)

        in_enc = torch.cat((prev_ref, code, next_ref), dim=1)
        
        # We give the entire shortcut input, even if some examples 
        # won't go through the shortcut, this is managed inside the conditional
        # coder forward. 
        # The conditional coder needs to have a <use_shortcut> vector of
        # dimension B, indicating wether it uses the shortcut or not (i.e.
        # replace the latents by some zeros.
        in_shortcut = torch.cat((prev_ref, next_ref), dim=1)

        net_input = {
            'in_enc': in_enc,
            'in_shortcut': in_shortcut,
            'flag_visu': flag_visu,
            'external_y': external_y,
            'idx_rate': idx_rate,
            'use_shortcut_vector': frame_type == FRAME_B,
            'generate_bitstream': generate_bitstream,
            'bitstream_path': bitstream_path,
            'frame_type': frame_type,
            'flag_bitstream_debug': flag_bitstream_debug,
            'latent_name': 'mofnet',
        }
        
        # Perform forward pass        
        raw_net_out, raw_visu = self.mode_net(net_input)

        # Some of ModeNet outputs need to be re-parameterized or slightly
        # modified
        B, C, H, W = code.size()
        decoded_y = raw_net_out.get('x_hat')[:, :, :H, :W]
        alpha = torch.clamp(decoded_y[:, 0, :, :].view(B, 1, H, W) + 0.5, 0., 1.)


        beta = torch.clamp(decoded_y[:, 1, :, :].view(B, 1, H, W) + 0.5, 0., 1.)
        v_prev = decoded_y[:, 2:4, :, :]
        v_next = decoded_y[:, 4:6, :, :]

        net_out = {
            'alpha': alpha,
            'beta': beta,
            'v_prev': v_prev,
            'v_next': v_next,
            'rate_y': raw_net_out.get('rate_y'),
            'rate_z': raw_net_out.get('rate_z'),
        }

        visu = {}
        if net_input.get('flag_visu'):
            TAG = 'ModeNet_'

            for k in raw_visu:
                visu[TAG + k] = raw_visu.get(k)

            visu[TAG + 'alpha'] = alpha
            visu[TAG + 'beta'] = beta
            visu[TAG + 'v_prev'] = v_prev
            visu[TAG + 'v_next'] = v_next
                
        return net_out, visu

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
