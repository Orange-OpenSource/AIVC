#/bin/bash
export HOME=/opt/GPU/Exp/hkzv2358/
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python encode.py -i ../raw_videos/BQMall_832x480_60_420/ --gop GOP_4 --end_frame 4 --start_frame 0 --model ms_ssim-5
python decode.py --model ms_ssim-5
python evaluate.py --raw ../raw_videos/BQMall_832x480_60_420/ --compressed ../compressed/
