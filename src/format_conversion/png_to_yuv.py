# Software Name: AIVC
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

import os
import subprocess

def png_to_yuv(in_dir, out_yuv, start_frame, end_frame, quiet=True):
    """
    Convert PNGs inside in_dir into a single .yuv file <out_yuv>.

    The PNGs whose indices are in [start_frame, end_frame] (included) are processed. 
    Other are ignored
    """
    # Path of the current module, used to obtain the absolute path of the .sh scripts
    current_module_path = os.path.dirname(os.path.realpath(__file__)) + '/'
    LJUST_SIZE = 40

    if not(in_dir.endswith('/')):
        in_dir += '/'

    if not(out_yuv.endswith('.yuv')):
        print('Output file must be a .yuv file')
        out_yuv += '.yuv'

    if not(quiet):
        print('[STATE] Starting png -> yuv conversion')

    # Better delete it as we're appending stuff
    os.system('rm ' + out_yuv)
    os.system('touch ' + out_yuv)
    nb_frame = end_frame - start_frame + 1

    for progress_cnt, i in enumerate(range(start_frame, end_frame + 1)):
        if not(quiet):
            msg = '\tFrame: '.ljust(LJUST_SIZE) + str(progress_cnt + 1).ljust(4) + '/ ' + str(nb_frame)
            print(msg, end='\r', flush=True)

        cmd = current_module_path + 'script_convert_one_frame/png_to_yuv.sh '
        cmd += in_dir + str(i) + ' '    # PNG inputs
        cmd += out_yuv + ' '
        cmd += in_dir + ' '
        cmd += current_module_path + 'script_convert_one_frame/convert_img.py'
        subprocess.call(cmd, shell=True)
