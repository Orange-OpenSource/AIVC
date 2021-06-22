import torch
from torch.nn import Module

from func_util.nn_util import get_value
from func_util.optical_flow import warp


class MotionCompensation(Module):

    def __init__(self, model_param):
        super(MotionCompensation, self).__init__()

        self.exact_p_frame = model_param.get('exact_p_frame')
        self.model_param = model_param

    def forward(self, param):
        """
        Perform the motion compensation. Two ref. frames are interpolated with two
        optical flows. Then, those 2 warping are combined with a pixel-wise
        weighting beta.
        """

        DEFAULT_PARAM = {
            # Previous reference frame x_{t-1}
            'prev': None,
            # Next reference x_{t+1}
            'next': None,
            # Optical flow to interpolate the previous frame into the frame to code
            'v_prev': None,
            # Optical flow to interpolate the next frame into the frame to code
            'v_next': None,
            # Pixel-wise ponderation between both interpolations
            'beta': None,
            # Interpolation mode
            'interpol_mode': 'bilinear',
            # Wether to output visualisation or not
            'flag_visu': False,
            # Which padding mode to use, available: 'zeros', 'border', 'reflection'
            'padding_mode': 'border',
            # Flag align_corners. Should be set to True every time
            'align_corners': True,
            # If true, we generate a bitstream at the end (and we don't go 
            # in the visu part?)
            'generate_bitstream': False,
            # Used to quantize the floatting grid in the warping function
            'quantizer': None,
        }

        # ===== RETRIEVE INPUTS ===== #
        prev_ref = get_value('prev', param, DEFAULT_PARAM)
        next_ref = get_value('next', param, DEFAULT_PARAM)
        v_prev = get_value('v_prev', param, DEFAULT_PARAM)
        v_next = get_value('v_next', param, DEFAULT_PARAM)
        beta = get_value('beta', param, DEFAULT_PARAM)
        interpol_mode = get_value('interpol_mode', param, DEFAULT_PARAM)
        flag_visu = get_value('flag_visu', param, DEFAULT_PARAM)
        padding_mode = get_value('padding_mode', param, DEFAULT_PARAM)
        align_corners = get_value('align_corners', param, DEFAULT_PARAM)
        generate_bitstream = get_value('generate_bitstream', param, DEFAULT_PARAM)
        quantizer = get_value('quantizer', param, DEFAULT_PARAM)
        # ===== RETRIEVE INPUTS ===== #

        net_out = {}

        if self.exact_p_frame:
            x_warp_from_prev = warp(
                prev_ref, v_prev, interpol_mode=interpol_mode, padding_mode=padding_mode, align_corners=align_corners
            )
            x_warp_from_next = torch.zeros_like(x_warp_from_prev, device=x_warp_from_prev.device)
            # Pixel-wise weighted combination
            net_out['x_warp'] = x_warp_from_prev
        else:
            # Perform both warpings
            x_warp_from_prev = warp(
                prev_ref, v_prev,
                interpol_mode=interpol_mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
            )
            x_warp_from_next = warp(
                next_ref, v_next,
                interpol_mode=interpol_mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
            )
            # Pixel-wise weighted combination
            net_out['x_warp'] = beta * x_warp_from_prev + (1 - beta) * x_warp_from_next

        visu = {}
        if flag_visu:
            TAG = 'MotionComp_'
            visu[TAG + 'png_warp_from_prev'] = x_warp_from_prev
            visu[TAG + 'png_warp_from_next'] = x_warp_from_next
            visu[TAG + 'png_warp_from_prev_masked'] = beta * x_warp_from_prev
            visu[TAG + 'png_warp_from_next_masked'] = (1 - beta) * x_warp_from_next



        return net_out, visu
