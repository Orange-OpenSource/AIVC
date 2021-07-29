# Software Name: AIVC
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

import argparse
import glob
import os
from clic21.metrics import evaluate

"""
Use the CLIC21 metrics to measure the quality on ALL the images contained in
the <compressed> arguments.

Print the size of the compressed file as well.
"""

# ============================ Arguments parsing ============================ #
parser = argparse.ArgumentParser()

parser.add_argument(
    '--raw', default='', type=str,
    help='Path of the folder containing the PNGs for the input data.'
)

parser.add_argument(
    '--compressed', default='', type=str,
    help='Path of the folder containing the compressed PNGs'
)

parser.add_argument(
    '--bitstream', default='../bitstream.bin', type=str, help='Path of the bitstream'
)
args = parser.parse_args()

path_raw = args.raw
if not(path_raw.endswith('/')):
    path_raw += '/'

path_compressed = args.compressed
if not(path_compressed.endswith('/')):
    path_compressed += '/'
# ============================ Arguments parsing ============================ #

# ========================= Evaluate the encoding =========================== #
# Get the indices of all images in path_compress
compressed_images_idx = [int(x.split('/')[-1].rstrip('_y.png')) for x in glob.glob(path_compressed + '*_y.png')]

target = {}
submit = {}
for idx in compressed_images_idx:
    for c in ['y', 'u', 'v']:
        cur_key = str(idx) + '_' + c
        submit[cur_key] = path_raw + str(idx) + '_' + c + '.png'
        target[cur_key] = path_compressed + str(idx) + '_' + c + '.png'


results = evaluate(submit, target)

print('PSNR    [dB]: ' + '%.5f' % (results.get('PSNR')))
print('MS-SSIM     : ' + '%.5f' % (results.get('MSSSIM')))
print('MS-SSIM [dB]: ' + '%.5f' % (results.get('MSSSIM_dB')))
# ========================= Evaluate the encoding =========================== #

# ===================== Print the size of the bitstream ===================== #
# Bitstream name is linked to the one of path_compressed
bitstream_path = args.bitstream
try:
    size_bytes = os.path.getsize(bitstream_path)
except FileNotFoundError:
    print('[ERROR]: bitstream not found, can not evaluate its size!')
    print('Bistream path: ' + bitstream_path)

print('Size [bytes]: ' + '%.0f' % (size_bytes))
# ===================== Print the size of the bitstream ===================== #
