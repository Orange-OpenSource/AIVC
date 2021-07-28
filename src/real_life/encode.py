# Software Name: AIVC
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

import os
import math
import time
import glob

from model_mngt.model_management import infer_one_sequence
from func_util.nn_util import get_value
from func_util.GOP_structure import GOP_STRUCT_DIC, get_gop_struct_name
from func_util.console_display import print_log_msg

def encode(param):
    DEFAULT_PARAM = {
        # The model to be evaluated. Must be a nn.Module
        'model': None,
        # Absolute path of the folder containing the PNGs.
        'sequence_path': '',
        # The GOP structure name, used only for logging, must be a string
        'GOP_struct_name': '',        
        # The GOP structure defined as in func_util/GOP_structure.py
        'GOP_struct': None,
        # For multi-rate
        'idx_rate': 0.,
        # Path of the final bitstream file
        'final_file': '',
        # Set to true to generate more stuff, useful for debug
        'flag_bitstream_debug': False,
        # First and last frame to encode (included)
        'idx_starting_frame': 1,
        # If set to -1: encode until the last frame
        'idx_end_frame': -1,
    }

    # =========================== RETRIEVE INPUTS =========================== #
    model = get_value('model', param, DEFAULT_PARAM)
    sequence_path = get_value('sequence_path', param, DEFAULT_PARAM)
    GOP_struct = get_value('GOP_struct', param, DEFAULT_PARAM)
    GOP_struct_name = get_value('GOP_struct_name', param, DEFAULT_PARAM)
    flag_bitstream_debug = get_value('flag_bitstream_debug', param, DEFAULT_PARAM)
    idx_rate = get_value('idx_rate', param, DEFAULT_PARAM)
    final_file = get_value('final_file', param, DEFAULT_PARAM)
    idx_starting_frame = get_value('idx_starting_frame', param, DEFAULT_PARAM)
    idx_end_frame = get_value('idx_end_frame', param, DEFAULT_PARAM)

    # =========================== RETRIEVE INPUTS =========================== #

    # Internal bitstream working dir
    bitstream_dir = './tmp_bitstream_working_dir/'

    if final_file == bitstream_dir.split('/')[-2]:
        print('ERROR: The bitstream file can not be in bitstream_dir')
        print('ERROR: Please change your directory!')
        return

    if not(sequence_path.endswith('/')):
        sequence_path += '/'

    # Count number of PNGs for the Y channel in sequence_path
    png_list = glob.glob(sequence_path + '*_y.png')
    png_idx = [int(x.split('/')[-1].rstrip('_y.png')) for x in png_list]
    max_index = max(png_idx)

    if (idx_starting_frame > idx_end_frame) and (idx_end_frame != -1):
        print('ERROR: First frame index bigger than last frame index')
        return
    # PNG numbering starts at 0!
    if idx_end_frame > max_index:
        print('ERROR: Last frame index exceeds the last frame index')
        return
    
    if idx_end_frame == -1:
        idx_end_frame = max_index

    # Clean bitstream directory
    os.system('rm -r ' + bitstream_dir)
    os.system('rm ' + bitstream_dir.rstrip('/'))
    os.system('mkdir -p ' + bitstream_dir)

    # Mainly use to output some debug values such as estimated rate, real-rate or PSNR
    encoder_out = {}
    working_dir = './logs/'
    print_log_msg('INFO', 'Start encoding', '', '')
    start_time = time.time()
    infer_one_sequence({
        'model': model,
        'GOP_struct_name': GOP_struct_name,        
        'GOP_struct': GOP_struct,
        'sequence_path': sequence_path,
        'idx_starting_frame': idx_starting_frame,
        'idx_end_frame': idx_end_frame,
        'idx_rate': idx_rate,
        'loading_mode': 'old',
        'bitstream_dir': bitstream_dir,
        'generate_bitstream': True,
        'flag_bitstream_debug': flag_bitstream_debug,
        'final_bitstream_path': final_file,
        'working_dir': working_dir,
    })
    # We're done for this sequence!
    elapsed_time = time.time() - start_time
    print_log_msg('INFO', 'Encoding done', '', '')
    print_log_msg('INFO', 'Bitstream path', '', final_file)

    # Measure the size of the data.zip file
    encoder_out = {}
    encoder_out['real_rate_byte'] = os.path.getsize(final_file)


    # Read log file to display some info
    result_file_name = working_dir + 'detailed.txt'
    f = open(result_file_name, 'r')

    # Last line = summary of the encoding
    line = f.readlines()[-1]
    # Parse line
    line = [x.lstrip(' ').rstrip(' ') for x in line.rstrip('\n').split('|')][1:-1]
    cur_seq_name = line[0]
    cur_psnr = float(line[2])
    cur_rate_bpp = float(line[3])
    cur_ms_ssim_db = float(line[9])
    cur_h = float(line[10])
    cur_w = float(line[11])

    # Number of frames we wanted to code.
    nb_frames_to_code = idx_end_frame - idx_starting_frame + 1
    # How many frames did we code in practice: add the padded frames.
    nb_coded_frames = math.ceil(nb_frames_to_code / len(GOP_struct)) * len(GOP_struct)
    # This is the estimated rate in byte
    cur_rate_byte = cur_rate_bpp * cur_h * cur_w * nb_coded_frames / 8

    encoder_out['psnr'] = cur_psnr
    encoder_out['ms_ssim_db'] = cur_ms_ssim_db
    encoder_out['h'] = cur_h
    encoder_out['w'] = cur_w
    encoder_out['nb_coded_frames'] = nb_coded_frames
    encoder_out['nb_frames_to_code'] = nb_frames_to_code
    encoder_out['nb_frames_gop'] = len(GOP_struct)
    encoder_out['estimated_rate_byte'] = cur_rate_byte
    rate_overhead = (encoder_out.get('real_rate_byte') / encoder_out.get('estimated_rate_byte') - 1) * 100        
    encoder_out['rate_overhead_percent'] = rate_overhead

    # Display the encoding results
    print_log_msg('INFO', 'Frame resolution', '[H x W]', str(int(encoder_out.get('h'))) + ' x ' + str(int(encoder_out.get('w'))))
    print_log_msg('INFO', 'First coded frame', '[frame]', int(idx_starting_frame))
    print_log_msg('INFO', 'Last coded frame', '[frame]', int(idx_end_frame))
    print_log_msg('INFO', 'Number of frames to code', '[frame]', int(encoder_out.get('nb_frames_to_code')))
    print_log_msg('INFO', 'Number of frames coded', '[frame]', int(encoder_out.get('nb_coded_frames')))
    print_log_msg('INFO', 'Intra-period', '[frame]', int(encoder_out.get('nb_frames_gop')))
    print_log_msg('RESULT', 'Number of frames', '[frame]', int(idx_end_frame - idx_starting_frame + 1))
    print_log_msg('RESULT', 'Encoding/decoding time', '[s]', '%.1f' % (elapsed_time))
    print_log_msg('RESULT', 'Encoding/decoding FPS', '[frame/s]', '%.1f' % ((idx_end_frame - idx_starting_frame + 1) / elapsed_time))
    print_log_msg('RESULT', 'Estimated PSNR', '[dB]', '%.4f' % (encoder_out.get('psnr')))
    print_log_msg('RESULT', 'Estimated MS-SSIM', '[dB]', '%.4f' % (encoder_out.get('ms_ssim_db')))
    print_log_msg('RESULT', 'Estimated rate', '[byte]', '%.1f' % (encoder_out.get('estimated_rate_byte')))
    print_log_msg('RESULT', 'Real rate', '[byte]', int(encoder_out.get('real_rate_byte')))
    print_log_msg('RESULT', 'Estimated rate overhead', '[%]', '%.2f' % (encoder_out.get('rate_overhead_percent')))

    # Clean the internal bitstream working dir
    os.system('rm -r ' + bitstream_dir)


    return encoder_out
