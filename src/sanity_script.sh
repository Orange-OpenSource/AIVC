#/bin/bash
export HOME=/opt/GPU/Exp/hkzv2358/
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python encode.py -i ../raw_videos/Cactus_1920x1080_50_420/ --gop GOP_32 --end_frame -1 --start_frame 0 --model ms_ssim-5
python decode.py --model ms_ssim-5
python evaluate.py --raw ../raw_videos/Cactus_1920x1080_50_420/ --compressed ../compressed/
