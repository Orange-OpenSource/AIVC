import torch

# Third-party modules
from torch.nn import Module

# Custom modules
from func_util.console_display import print_log_msg
from func_util.nn_util import get_value
from func_util.GOP_structure import FRAME_I
from models.base_ae.conditional_ae import ConditionalNet


class CodecNet(Module):

    def __init__(self, model_param):

        super(CodecNet, self).__init__()

        # MofNet main encoder has 6 inputs (2 frames).
        # Mofnet encoder shortcut has 3 inputs (1 frame)
        # Mofnet main decoder has 3 outputs (1 frame)
        model_param['in_c'] = 6
        model_param['out_c'] = 3
        model_param['in_c_shortcut_y'] = 3
        self.codec_net = ConditionalNet(model_param)

        # ===== LOG MESSAGES ===== #
        print_log_msg('DEBUG', '__init__ CodecNet', 'state', 'done')
        # ===== LOG MESSAGES ===== #

    def forward(self, param):
        DEFAULT_PARAM = {
            # Tensor to encode/decode
            'code': None,
            # Predicted frame \tilde{x}_t
            'prediction': None,
            # If not None: override the y with external y
            'external_y': None,
            # For multi-rate, not used for now
            'idx_rate': 0.,
            # A scalar, indicating the type of the frame for
            # each of the B examples which are either: FRAME_I, FRAME_P or 
            # FRAME_B
            'frame_type': None,
            # If true, we generate a bitstream at the end
            'generate_bitstream': False,
            # Path where the bistream is written
            'bitstream_path': '',
            # Set to true to generate more stuff, useful for debug
            'flag_bitstream_debug': False,
        }

        # Retrieve mode_net input data
        code = get_value('code', param, DEFAULT_PARAM)
        prediction = get_value('prediction', param, DEFAULT_PARAM)
        external_y = get_value('external_y', param, DEFAULT_PARAM)
        idx_rate = get_value('idx_rate', param, DEFAULT_PARAM)
        frame_type = get_value('frame_type', param, DEFAULT_PARAM)
        generate_bitstream = get_value('generate_bitstream', param, DEFAULT_PARAM)
        bitstream_path = get_value('bitstream_path', param, DEFAULT_PARAM)
        flag_bitstream_debug = get_value('flag_bitstream_debug', param, DEFAULT_PARAM)

        net_input = {
            'in_enc': torch.cat((prediction, code), dim=1),
            'in_shortcut': prediction,
            'external_y': external_y,
            'idx_rate': idx_rate,
            'use_shortcut_vector': frame_type != FRAME_I,
            'frame_type': frame_type,
            'generate_bitstream': generate_bitstream,
            'bitstream_path': bitstream_path,
            'flag_bitstream_debug': flag_bitstream_debug,
            'latent_name': 'codecnet'
        }

        raw_net_out = self.codec_net(net_input)

        # Some of CodecNet outputs need to be re-parameterized or slightly
        # modified
        B, C, H, W = code.size()

        net_out = {
            'x_hat': raw_net_out.get('x_hat')[:, :, :H, :W],
            'rate_y': raw_net_out.get('rate_y'),
            'rate_z': raw_net_out.get('rate_z'),
        }

        return net_out
