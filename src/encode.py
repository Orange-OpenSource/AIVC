# Software Name: AIVC
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

import argparse
import subprocess
import torch
import glob
import os
import torch.quantization

from format_conversion.yuv_to_png import yuv_to_png
from func_util.cluster_mngt import set_compute_param, COMPUTE_PARAM, seed_all
from func_util.GOP_structure import GOP_STRUCT_DIC
from model_mngt.model_management import load_model
from real_life.encode import encode
from real_life.decode import Decoder

# ! Must be launched from within /src

# ============================ Arguments parsing ============================ #
parser = argparse.ArgumentParser()

parser.add_argument(
    '--gop', default='GOP_32', type=str,
    help='Name of the GOP structure to be used. '
)

parser.add_argument(
    '--model', default='ms_ssim-1', type=str,
    help='Name of the pre-trained model. Either ms_ssim-x or psnr-x. ' +
    'x ranges from 1 (high rate) to 7 (low rate).'
)

parser.add_argument(
    '-i', default='../raw_videos/BQMall_832x480_60_420/', type=str,
    help='Path of the input data. It can be either a .yuv file, or a ' +
    'directory with PNGs triplets in it.',
)

parser.add_argument(
    '-o', default='../bitstream.bin', type=str, help='Path of the output bitstream',
)

parser.add_argument(
    '--start_frame', default=0, type=int, help='Index of the first frame to encode.'
)

parser.add_argument(
    '--end_frame', default=-1, type=int,
    help='Index of the last frame to encode (included). If equals to -1: ' +
    'process until the last frame.'
)

parser.add_argument(
    '--rng_seed', default=666, type=int,
    help='RNG seed, must be set identically between encoder and decoder.'
)

parser.add_argument('--cpu', help='Run on cpu', action='store_true')
args = parser.parse_args()
# ============================ Arguments parsing ============================ #

# =========================== Data pre-processing =========================== #
flag_yuv = args.i.endswith('.yuv')

# We need to convert the data to PNG triplets.
if flag_yuv:
    seq_path = args.i.rstrip('.yuv') + '/'
    yuv_to_png(args.i, seq_path, args.start_frame, args.end_frame, check_lossless=False)
    
else:
    seq_path = args.i
    if not(seq_path.endswith('/')):
        seq_path += '/'
# =========================== Data pre-processing =========================== #

# ========================= Compute param arguments ========================= #
if args.cpu:
    flag_gpu = False
    device = 'cpu'
else:
    flag_gpu = True
    device = 'cuda:0'
    
set_compute_param('flag_gpu', flag_gpu)
set_compute_param('device', device)
set_compute_param('workers', 1)
# ! This is absolutely necessary to ensure proper arithmetic encoding/decoding
seed_all(seed=args.rng_seed)
# ========================= Compute param arguments ========================= #


# ======================== Load the complete system ========================= #
# Set working dir to the correct model
os.chdir('../models/' + args.model + '/')
# Load last save
model = load_model(prefix='0_', on_cpu=True).to(COMPUTE_PARAM.get('device')).eval()
# Set working dir back to /src/
os.chdir('../../src/')
# ======================== Load the complete system ========================= #

# ========================== Perform the encoding =========================== #
encode({
    'model': model,
    'sequence_path': seq_path,
    'GOP_struct': GOP_STRUCT_DIC.get(args.gop),
    'GOP_struct_name': args.gop,
    'idx_rate': 0,
    'final_file': args.o,
    'idx_starting_frame': args.start_frame,
    'idx_end_frame': args.end_frame,
})
# ========================== Perform the encoding =========================== #
