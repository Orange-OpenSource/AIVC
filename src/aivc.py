# Software Name: AIVC
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

import argparse
import sys
import subprocess
import os

# ============================ Arguments parsing ============================ #
parser = argparse.ArgumentParser()

parser.add_argument(
    '--coding_config', default='RA', type=str,
    help='Available:\n\t- "RA" for random access\n\t- "LDP" for low-delay P' +
    '\n\t- "AI" for all intra '
)

parser.add_argument(
    '--gop_size', default=32, type=int,
    help='Only for RA coding. Size of the hierarchical GOP. Min is 2. ' +
    'Must be less or equal to the intra period.' +
    'Must be a power of 2 for RA and coding.'
)

parser.add_argument(
    '--intra_period', default=32, type=int,
    help='Number of inter frames in between two intra frames. Min is 2. ' +
    'This is ignored for intra coding' +
    'Must be a multiple of GOP size for RA coding.'
)

parser.add_argument(
    '--model', default='ms_ssim-2021cc-3', type=str,
    help='Name of the pre-trained model: ms_ssim-2021cc-x ' +
    'x ranges from 1 (high rate) to 7 (low rate).'
)

parser.add_argument(
    '-i', default=' ../raw_videos/BlowingBubbles_416x240_50_420/', type=str,
    help='Path of the input data. It can be either a .yuv file, or a ' +
    'directory with PNGs triplets in it.',
)

parser.add_argument(
    '--start_frame', default=0, type=int, help='Index of the first frame to encode.'
)

parser.add_argument(
    '--end_frame', default=90, type=int,
    help='Index of the last frame to encode (included). If equals to -1: ' +
    'process until the last frame.'
)

parser.add_argument(
    '--bitstream_out', default='../bitstream.bin', type=str,
    help='Path of the output bitstream.',
)

parser.add_argument(
    '-o', default='../compressed.yuv', type=str,
    help='Path of the output compressed video.'
)

parser.add_argument(
    '--rng_seed', default=666, type=int,
    help='RNG seed, must be set identically between encoder and decoder.'
)

parser.add_argument('--cpu', help='Run on cpu', action='store_true')
args = parser.parse_args()
# ============================ Arguments parsing ============================ #

# =================== Parse the desired coding structure ==================== #
if args.coding_config == 'AI':
    gop = '1_GOP_0'

elif args.coding_config == 'LDP':
    if args.intra_period not in range(2, 65535):
        print('[ERROR]: Intra period should be in [2, 65535] for LDP.')
        sys.exit(1)
    gop = f'LDP_{args.intra_period}'

elif args.coding_config == 'RA':
    nb_gop = args.intra_period / args.gop_size
    if nb_gop % 1 != 0:
        print('[ERROR]: Intra period must be equal a multiple of GOP size')
        sys.exit(1)
    nb_gop = int(nb_gop)

    if nb_gop not in range(1, 65535):
        print('[ERROR]: Intra period must be equal to [1, 65535] times GOP size')
        sys.exit(1)
    if args.gop_size not in range(2, 65535):
        print('[ERROR]: Intra period must be in [1, 65535]')
        sys.exit(1)

    gop = f'{nb_gop}_GOP_{args.gop_size}'

else:
    print('[ERROR]: unknown coding configuration. Should be either RA, AI or LDP.')
    sys.exit(1)
# =================== Parse the desired coding structure ==================== #

# ================ Perform encoding, decoding and evaluation ================ #
# Needed for reproducibility between encoding or decoding
os.system('export CUBLAS_WORKSPACE_CONFIG=:4096:8')

# Perform encoding
print(('*' * 80).center(120))
print('Starting encoding'.center(120))
cmd = 'python encode.py -i ' + args.i + ' --gop ' + gop + ' --model ' + args.model
cmd += ' --start_frame ' + str(args.start_frame) + ' --end_frame ' + str(args.end_frame) 
cmd += ' -o ' + args.bitstream_out + ' --rng_seed ' + str(args.rng_seed)
if args.cpu:
    cmd += ' --cpu'
subprocess.call(cmd, shell=True)

# Perform decoding
print(('*' * 80).center(120))
print('Starting decoding'.center(120))
cmd = 'python decode.py --model ' + args.model + ' -i ' + args.bitstream_out
cmd += ' -o ' + args.o + ' --rng_seed ' + str(args.rng_seed)
if args.cpu:
    cmd += ' --cpu'
subprocess.call(cmd, shell=True)

# Evaluate
print(('*' * 80).center(120))
print('Starting evaluation'.center(120))
cmd = 'python evaluate.py --raw ' + args.i.rstrip('.yuv').rstrip('/') + '/'
cmd += ' --compressed ' + args.o.rstrip('.yuv').rstrip('/') + '/'
cmd += ' --bitstream ' + args.bitstream_out
subprocess.call(cmd, shell=True)
# ================ Perform encoding, decoding and evaluation ================ #

