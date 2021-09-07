# Software Name: AIVC
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

"""
Module gathering some functions which ease the logging of results
inside a file. All functions return strings which can then 
be printed in a file or in a console.
"""

# Hard constants are defined here. It is only about printing so it 
# doesn't matter if it is hard coded.

SIZE_PIC_NAME = 40
SIZE_COL_LOG = 12

def generate_header_file():
    """
    Generate the top row of a log file.
    """
    msg = '|' + 'Video'.center(SIZE_PIC_NAME)
    msg += '|' + 'Frame idx.'.center(SIZE_COL_LOG)
    msg += '|' + 'PSNR dB'.center(SIZE_COL_LOG)
    msg += '|' + 'R bpp'.center(SIZE_COL_LOG)
    msg += '|' + 'R Mode bpp'.center(SIZE_COL_LOG)
    msg += '|' + 'R Codec bpp'.center(SIZE_COL_LOG)
    msg += '|' + 'alpha'.center(SIZE_COL_LOG)
    msg += '|' + 'beta'.center(SIZE_COL_LOG)
    msg += '|' + 'Loss'.center(SIZE_COL_LOG)
    msg += '|' + 'MS-SSIM dB'.center(SIZE_COL_LOG)
    msg += '|' + 'h'.center(SIZE_COL_LOG)
    msg += '|' + 'w'.center(SIZE_COL_LOG)
    msg += '|\n'

    return msg

def generate_log_metric_one_frame(result):
    """
    From a result dictionnary of the frame, generate a string in order to
    log the result of this frame.
    """

    msg = '|' + str(result.get('pic_name')).center(SIZE_PIC_NAME)
    msg += '|' + str(result.get('frame_idx')).center(SIZE_COL_LOG)
    msg += '|' + ('%.5f' % (result.get('psnr'))).center(SIZE_COL_LOG)
    msg += '|' + ('%.6f' % (result.get('total_rate_bpp'))).center(SIZE_COL_LOG)
    msg += '|' + ('%.6f' % (result.get('mode_rate_bpp'))).center(SIZE_COL_LOG)
    msg += '|' + ('%.6f' % (result.get('codec_rate_bpp'))).center(SIZE_COL_LOG)
    msg += '|' + ('%.3f' % (result.get('mean_alpha'))).center(SIZE_COL_LOG)
    msg += '|' + ('%.3f' % (result.get('mean_beta'))).center(SIZE_COL_LOG)
    msg += '|' + ('%.5f' % (result.get('loss'))).center(SIZE_COL_LOG)
    msg += '|' + ('%.5f' % (result.get('ms_ssim_db'))).center(SIZE_COL_LOG)
    msg += '|' + str(result.get('h').item()).center(SIZE_COL_LOG)
    msg += '|' + str(result.get('w').item()).center(SIZE_COL_LOG)
    msg += '|\n'
    return msg
