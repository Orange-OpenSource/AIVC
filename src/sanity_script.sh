#/bin/bash
# export HOME=/opt/GPU/Exp/hkzv2358/
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# python encode.py -i ../raw_videos/BlowingBubbles_416x240_50_420.yuv --gop GOP_32 --end_frame 100 --start_frame 0 --model ms_ssim-5
# python decode.py --model ms_ssim-5
python evaluate.py --raw ../raw_videos/BlowingBubbles_416x240_50_420/ --compressed ../compressed/
