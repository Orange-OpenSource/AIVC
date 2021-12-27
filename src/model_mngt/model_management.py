# Software Name: AIVC
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

"""
This module contains all the methods needed to train a model:
    - Training
    - Test
    - Load and save a model
    - Setting a model to training of evaluation mode
"""

import os
import torch
import math

from func_util.console_display import print_log_msg
from func_util.img_processing import load_frames, cast_before_png_saving
from func_util.nn_util import crop_dic, get_value
from func_util.result_logging import generate_header_file, generate_log_metric_one_frame
from model_mngt.loss_function import compute_metrics_one_GOP, average_N_frame
from real_life.cat_binary_files import cat_one_video
from real_life.bitstream import ArithmeticCoder


def infer_one_sequence(param):
    """
    The purpose of this function is to infer N successive GOPs from one
    sequences. It relies on the function infer_one_GOP to generate the output
    data and results.
    """

    DEFAULT_PARAM = {
        # The model to be trained. Must be a nn.Module
        'model': None,
        # The GOP structure defined as in func_util/GOP_structure.py
        'GOP_struct': None,
        # The name of the GOP structure, mainly for logging purpose
        'GOP_struct_name': None,
        # Absolute path of the folder containing the 3N PNG of the sequences
        # (if YUV) of the N PNG if RGB.
        'sequence_path': '',
        # What is the first frame we encode. The previous one are skipped. 0 means
        # we start at the first frame, 1 at the second etc.
        'idx_starting_frame': 0,
        # We want to compress frame from <idx_starting_frame> to <idx_end_frame> **included**
        'idx_end_frame': 8,
        # If True, there is only one PNG per frame, with the 3 color channels in it.
        # If False, each frame needs 3 PNGs to be described: Y, U, and V.
        'rgb': False,
        # How the (raw) video frames are organized. Available:
        #   <old> : used for the ICLR paper i.e. <sequence_path>/idx_<y,u,v>.png
        #   <clic>: <sequence_path>/<sequence_name>_<padded_idx>_<y,u,v>.png
        'loading_mode': 'old',
        # If true, we generate a bitstream at the end
        'generate_bitstream': False,
        # Path of the directory in which we output the bitstream
        'bitstream_dir': '',
        # For multi-rate
        'idx_rate': 0.,
        # Set to true to generate more stuff, useful for debug
        'flag_bitstream_debug': False,
        # All internal log files for the NN will be written in this directory
        'working_dir': '../logs/',
        # Path of the final bitstream file
        'final_bitstream_path': '',
    }

    # ========== RETRIEVE INPUTS ========== #
    model = get_value('model', param, DEFAULT_PARAM)
    GOP_struct = get_value('GOP_struct', param, DEFAULT_PARAM)
    GOP_struct_name = get_value('GOP_struct_name', param, DEFAULT_PARAM)
    sequence_path = get_value('sequence_path', param, DEFAULT_PARAM)
    # nb_GOP = get_value('nb_GOP', param, DEFAULT_PARAM)
    idx_starting_frame = get_value('idx_starting_frame', param, DEFAULT_PARAM)
    idx_end_frame = get_value('idx_end_frame', param, DEFAULT_PARAM)
    rgb = get_value('rgb', param, DEFAULT_PARAM)
    loading_mode = get_value('loading_mode', param, DEFAULT_PARAM)
    generate_bitstream = get_value('generate_bitstream', param, DEFAULT_PARAM)
    bitstream_dir = get_value('bitstream_dir', param, DEFAULT_PARAM)
    idx_rate = get_value('idx_rate', param, DEFAULT_PARAM)
    flag_bitstream_debug = get_value('flag_bitstream_debug', param, DEFAULT_PARAM)
    working_dir = get_value('working_dir', param, DEFAULT_PARAM)
    final_bitstream_path = get_value('final_bitstream_path', param, DEFAULT_PARAM)
    # ========== RETRIEVE INPUTS ========== #

    # ========== IT WAS IN THE TEST FUNCTION ============= #
    if not(working_dir.endswith('/')):
        working_dir += '/'

    # Retrieve lambda, useful to compute some losses
    lambda_tradeoff_list = model.model_param.get('lambda_tradeoff')

    # Construct working dir
    os.system('mkdir -p ' + working_dir)

    # # Clean pre-existing result files
    # result_file_name = working_dir + 'summary.txt'
    # # result_file_name = working_dir + 'results_' + str(GOP_struct_name) + '_rate_' + str(idx_rate).replace('.', '-') + '.txt'
    # os.system('rm ' + result_file_name)

    # # Create log file, this log file is here for all sequences
    # file_res = open(result_file_name, 'w')
    # file_res.write(generate_header_file())

    # Lambda trade-off is only useful to compute some losses in the metric
    # ! For now, round idx_rate to select the lambda trade_off
    rounded_idx_rate = int(round(idx_rate))
    lambda_tradeoff = lambda_tradeoff_list[rounded_idx_rate]
    l_codec = lambda_tradeoff
    l_mof = lambda_tradeoff

    sequence_name = sequence_path.split('/')[-2]
    print_log_msg('INFO', 'infer_one_sequence', 'sequence name', sequence_name)
    print_log_msg('INFO', 'infer_one_sequence', 'GOP_struct_name', GOP_struct_name)
    print_log_msg('INFO', 'infer_one_sequence', 'idx_rate', idx_rate)
    # ========== IT WAS IN THE TEST FUNCTION ============= #

    # ========== COMPUTE NUMBER OF FRAMES & GOP ========== #
    if generate_bitstream:
        if not(bitstream_dir.endswith('/')):
            bitstream_dir += '/'

        os.system('rm -r ' + bitstream_dir)
        os.system('mkdir -p ' + bitstream_dir)

        if flag_bitstream_debug:
            debug_dir = '.' + '/'.join(bitstream_dir.split('/')[:-3]) + '/debug/' + bitstream_dir.split('/')[-2] + '/'
            os.system('rm ' + debug_dir + '*.md5')
            os.system('mkdir -p ' + debug_dir)

    # Number of frames in the video sequence
    if not(sequence_path.endswith('/')):
        sequence_path += '/'

    # How many frames we have to compress
    nb_frames = idx_end_frame - idx_starting_frame + 1
    # How many frames in a GOP
    GOP_size = len(GOP_struct)
    # How many GOPs we have to process. If there is a uncomplete GOP
    # e.g. 5 frames in the sequences and a  GOP size of 3, do 2 GOPs
    # and pad the last one.
    nb_GOP = math.ceil(nb_frames / GOP_size)

    # How many frames are required for our number of GOPs
    expected_nb_frames = nb_GOP * GOP_size
    # When loading the last GOP, we'll need to pad the last GOP by <nb_frame_to_pad>
    nb_frame_to_pad = expected_nb_frames - nb_frames
    # ========== COMPUTE NUMBER OF FRAMES & GOP ========== #

    # These dictionaries are here to log the result on the entire sequences.
    sequence_result = {}

    print_log_msg('DEBUG', 'infer_one_sequence', 'nb_GOP', nb_GOP)

    for i in range(nb_GOP):
        # Except for the last GOP, we don't need any padding
        if i == nb_GOP - 1:
            cur_nb_frame_to_pad = nb_frame_to_pad
        else:
            cur_nb_frame_to_pad = 0

        # Load the frames
        raw_frames = load_frames({
            'sequence_path': sequence_path,
            'idx_starting_frame': i * GOP_size + idx_starting_frame,
            'nb_frame_to_load': GOP_size,
            'nb_pad_frame': cur_nb_frame_to_pad,
            'rgb': rgb,
            'loading_mode': loading_mode,
        })

        # Perform forward for this GOP
        _, GOP_result = infer_one_GOP({
            'model': model,
            'GOP_struct': GOP_struct,
            'raw_frames': raw_frames,
            'l_codec': l_codec,
            'l_mof': l_mof,
            'index_GOP_in_video': i,
            'generate_bitstream': generate_bitstream,
            'bitstream_dir': bitstream_dir,
            # Same as idx_starting_frame in load_frames above
            'real_idx_first_frame': i * GOP_size + idx_starting_frame,
            'idx_rate': idx_rate,
            'flag_bitstream_debug': flag_bitstream_debug,
        })

        # Retrieve some results for this GOP. To spare some memory, we don't
        # keep trace of everything in net_out
        # Beware, we do not retrieve GOP average result i.e. 'GOP' entry.
        for f in range(GOP_size):
            # Name of the frame inside the GOP
            GOP_frame_name = 'frame_' + str(f)
            # Name of the frame in the entire sequence
            seq_frame_name = 'frame_' + str(i * GOP_size + f + idx_starting_frame)

            # Retrieve all results from the current frame
            sequence_result[seq_frame_name] = GOP_result.get(GOP_frame_name)

    # We're almost done, we just have to average the results from the N frame
    # to obtain the 'sequence' entry of the sequence_result dictionnary.
    # This is pretty much the same thing as what is done for a GOP averaging in the
    # loss function
    sequence_result['sequence'] = average_N_frame({
        'x': sequence_result,
        'nb_pad_frame': cur_nb_frame_to_pad,
    })

    if generate_bitstream:
        # input('Before cat one video')
        cat_one_video({
            'bitstream_dir': bitstream_dir,
            'idx_starting_frame': idx_starting_frame,
            'idx_end_frame': idx_end_frame,
            'final_bitstream_path': final_bitstream_path,
            })
        # input('After cat one video')

    # ========== IT WAS IN THE TEST FUNCTION ============= #
    # Log all metrics for all frames in a special log file
    # Special repositories for the sequences detailed log file
    file_res_sequence = open(working_dir + 'detailed.txt', 'w')
    file_res_sequence.write(generate_header_file())

    for f in sequence_result:
        sequence_result[f]['pic_name'] = sequence_name
        sequence_result[f]['frame_idx'] = f
        file_res_sequence.write(generate_log_metric_one_frame(sequence_result.get(f)))

    # # Add a final line in the logging for the entire sequence in the general file
    # file_res.write(generate_log_metric_one_frame(sequence_result.get('sequence')))

    file_res_sequence.close()
    # ========== IT WAS IN THE TEST FUNCTION ============= #

    # We're done we can return the sequence result
    return sequence_result


def infer_one_GOP(param):
    """
    The purpose of this function is to infer one GOP and to
    return a result dictionnary, gathering as much metrics as needed.

    This is separate from the training function, and it is basically
    a warper around the GOP_forward function of the model.
    """
    DEFAULT_PARAM = {
        # The model to be trained. Must be a nn.Module
        'model': None,
        # The GOP structure defined as in func_util/GOP_structure.py
        'GOP_struct': None,
        # The uncompressed frames (i.e. the frames to code), defined as:
        #   frame_0: {'y': tensor, 'u': tensor, 'v': tensor}
        #   frame_1: {'y': tensor, 'u': tensor, 'v': tensor}
        'raw_frames': None,
        # Lambda for CodecNet rate
        'l_codec': 0.,
        # Lambda for MOFNet rate
        'l_mof': 0.,
        # Index of the GOP in the video. Scalar in [0, N]
        'index_GOP_in_video': 0,
        # If true, we generate a bitstream at the end
        'generate_bitstream': False,
        # Path of the directory in which we output the bitstream
        'bitstream_dir': '',
        # Frame index in the video of the first frame (I) of the
        # GOP.
        'real_idx_first_frame': 0,
        # For multi-rate
        'idx_rate': 0.,
        # Set to true to generate more stuff, useful for debug
        'flag_bitstream_debug': False,
    }

    # ========== RETRIEVE INPUTS ========== #
    model = get_value('model', param, DEFAULT_PARAM)
    GOP_struct = get_value('GOP_struct', param, DEFAULT_PARAM)
    raw_frames = get_value('raw_frames', param, DEFAULT_PARAM)
    l_codec = get_value('l_codec', param, DEFAULT_PARAM)
    l_mof = get_value('l_mof', param, DEFAULT_PARAM)
    index_GOP_in_video = get_value('index_GOP_in_video', param, DEFAULT_PARAM)
    generate_bitstream = get_value('generate_bitstream', param, DEFAULT_PARAM)
    bitstream_dir = get_value('bitstream_dir', param, DEFAULT_PARAM)
    real_idx_first_frame = get_value('real_idx_first_frame', param, DEFAULT_PARAM)
    idx_rate = get_value('idx_rate', param, DEFAULT_PARAM)
    flag_bitstream_debug = get_value('flag_bitstream_debug', param, DEFAULT_PARAM)
    # ========== RETRIEVE INPUTS ========== #

    # Set model to evaluation mode
    model = model.eval()
    # # Retrieve the device on which we're working
    # my_device = COMPUTE_PARAM.get('device')
    # # Data represent the raw frames (i.e. uncompressed)
    # raw_frames = push_gop_to_device(raw_frames, my_device)

    model_input = {
        'GOP_struct': GOP_struct,
        'raw_frames': raw_frames,
        'idx_rate': idx_rate,
        'index_GOP_in_video': index_GOP_in_video,
        'generate_bitstream': generate_bitstream,
        'real_idx_first_frame': real_idx_first_frame,
        'bitstream_dir': bitstream_dir,
        'flag_bitstream_debug': flag_bitstream_debug,
    }

    with torch.no_grad():
        net_out = model.GOP_forward(model_input)
        for f in net_out:
            # Clamp and crop the output
            net_out[f]['x_hat'] = crop_dic(net_out.get(f).get('x_hat'), raw_frames.get(f))
            net_out[f]['x_hat'] = cast_before_png_saving({
                'x': net_out.get(f).get('x_hat'), 'data_type': 'yuv_dic',
            })

    torch.cuda.empty_cache()
    _, result = compute_metrics_one_GOP({
        'net_out': net_out,
        'target': raw_frames,
        'l_mof': l_mof,
        'l_codec': l_codec,
    })

    # net_out should not be that useful,;and results contains a bunch of metrics
    # for the different frames.
    return net_out, result


def load_model(prefix='', on_cpu=False):
    """
    Load the model with a given prefix as parameter
    """

    map_loc = torch.device('cpu') if on_cpu else None
    model = torch.load('./' + prefix + 'model.pt', map_location=map_loc)

    # We add it to the class atribute. This way it is save alongside
    # the model and can be loaded in the decoder
    model.codec_net.codec_net.ac = ArithmeticCoder({
        'balle_pdf_estim_z': model.codec_net.codec_net.pdf_z,
        'device': map_loc,
    })

    model.mode_net.mode_net.ac = ArithmeticCoder({
        'balle_pdf_estim_z': model.mode_net.mode_net.pdf_z,
        'device': map_loc,
    })

    return model
