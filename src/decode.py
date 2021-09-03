# Software Name: AIVC
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

import os
import torch
import argparse
import glob

from format_conversion.png_to_yuv import png_to_yuv
from func_util.console_display import print_log_msg
from func_util.cluster_mngt import set_compute_param, COMPUTE_PARAM, seed_all
from real_life.decode import decode_one_video, Decoder
from model_mngt.model_management import load_model

# ! Must be launched from within /src

# ============================ Arguments parsing ============================ #
parser = argparse.ArgumentParser()
parser.add_argument('--cpu', help='Run on cpu', action='store_true')
parser.add_argument('-i', default='../bitstream.bin', type=str, help='Bitstream path')
parser.add_argument('-o', default='../compressed.yuv', type=str, help='Output file')

parser.add_argument(
    '--model', default='ms_ssim-1', type=str,
    help='Name of the pre-trained model. Either ms_ssim-x or psnr-x. ' +
    'x ranges from 1 (high rate) to 7 (low rate).'
)

parser.add_argument(
    '--rng_seed', default=666, type=int,
    help='RNG seed, must be set identically between encoder and decoder.'
)

args = parser.parse_args()

if not(args.o.endswith('.yuv')):
    out_file = args.o + '.yuv'
else:
    out_file = args.o

# out_dir is for all the PNGs,
# out_file is for the compressed .yuv
out_dir = out_file.rstrip('.yuv') + '/'
# ============================ Arguments parsing ============================ #

# ============================ Load the decoder ============================= #
# Set working dir to the correct model
os.chdir('../models/' + args.model + '/')
# Load last save
model = load_model(prefix='0_', on_cpu=True).to(COMPUTE_PARAM.get('device')).eval()
# Set working dir back to /src/
os.chdir('../../src/')
# Construct the decoder
my_decoder = Decoder({'full_net': model, 'device': COMPUTE_PARAM.get('device')}).eval()
# ============================ Load the decoder ============================= #

# ========================== Manage cuda parameters ========================= #
if torch.cuda.is_available() and not(args.cpu):
    flag_gpu = True
    device = 'cuda:0'
    print_log_msg('INFO', '', '', 'Found CUDA device')    
else:
    flag_gpu = False
    device = 'cpu'
    print_log_msg('INFO', '', '', 'No CUDA device available')


set_compute_param('flag_gpu', flag_gpu)
set_compute_param('device', device)
# ! This is absolutely necessary to ensure proper arithmetic encoding/decoding
seed_all(seed=args.rng_seed)

my_decoder = my_decoder.to(COMPUTE_PARAM.get('device'))
# ========================== Manage cuda parameters ========================= #

# ========================== Perform the decoding =========================== #
decode_one_video({
    # Decoder with which we're going to decode the GOP
    'decoder': my_decoder,
    # Absolute path of the data.zip file
    'bitstream_path': args.i,
    # On which device the decoder runs
    'device': COMPUTE_PARAM.get('device'),
    # Folder in which we output the decoded frames.
    'out_dir': out_dir,
})
# ========================== Perform the decoding =========================== #

# =========================== Convert PNG to YUV ============================ #
# Get index of the decoded frame.
list_png_idx = [int(x.split('/')[-1].rstrip('_y.png')) for x in glob.glob(out_dir + '*_y.png')]

# Convert all the decoded PNGs in out_dir into a single .yuv file called out_file
png_to_yuv(out_dir, out_file, min(list_png_idx), max(list_png_idx), quiet=False)
print_log_msg('INFO', 'Final decoded video', '', out_file)
# =========================== Convert PNG to YUV ============================ #
