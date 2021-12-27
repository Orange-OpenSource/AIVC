#!/bin/bash
# Required to ensure reproducibility between encoding and decoding
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python aivc.py \
    -i ../raw_videos/BlowingBubbles_416x240_50_420.yuv \
    -o ../compressed.yuv \
    --bitstream_out ../bitstream.bin \
    --start_frame 0 \
    --end_frame 100 \
    --coding_config RA \
    --gop_size 16 \
    --intra_period 32 \
    --model ms_ssim-2021cc-6