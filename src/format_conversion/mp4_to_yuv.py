# Software Name: AIVC
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

"""
Convert one mp4 video file to a yuv video file (420)
"""
import argparse
import os
from utils import get_video_resolution, get_video_fps


parser = argparse.ArgumentParser()

parser.add_argument(
    '-i',
    '--input',
    help='Absolute path of the mp4 video file to be converted.' +
    ' Must end with .mp4',
    type=str,
)

parser.add_argument(
    '-o',
    '--output',
    help='Absolute path of the output directory. The .yuv file created ' +
    'inside will be called converted_widthxheight_fps_420.yuv',
    type=str,
)

parser.add_argument(
    '--quiet',
    help='When used, nothing is printed',
    action='store_true'
)

args = parser.parse_args()

in_file = args.input
out_dir = args.output

LJUST_SIZE = 40

# Printing file name and checking the extension
if not(in_file.endswith('.mp4')):
    print('[ERROR] Input file is not a mp4 file')

if not(out_dir.endswith('/')):
    out_dir += '/'

if not(args.quiet):
    print('[STATE] Start processing: mp4 -> yuv')
    print('\tInput file: '.ljust(LJUST_SIZE) + in_file)
    print('\tOutput directory: '.ljust(LJUST_SIZE) + out_dir)

# Create the directory in which the output file will be created
# Just in case is does not exist
os.system('mkdir -p ' + out_dir)
# Compute absolute path of the output file. For this, we need to have
# the width and height of the video as well as the fps
w, h = get_video_resolution(in_file)
fps = get_video_fps(in_file)
# Only the name, no path, no extension
yuv_file_suffix = '_' + str(int(w)) + 'x' + str(int(h)) + '_' + str(int(fps)) + '_420.yuv'
out_yuv_name = out_dir + 'converted' + yuv_file_suffix

# Convert MP4 to raw yuv 420 planar video
cmd = 'ffmpeg -i ' + in_file
cmd += ' -c:v rawvideo -pixel_format yuv420p '
cmd += out_yuv_name + ' > /dev/null 2>&1'
os.system(cmd)
if not(args.quiet):
    print('[STATE] End processing: mp4 -> yuv')
