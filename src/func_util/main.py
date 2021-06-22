"""
* 1. Number of examples

The dataset is prepared to work with maximal GOP size of 32, so we need
to prepare NB_FRAMES = 32 + 1 frames successive frame.
# ! NB_FRAMES = MAX_GOP_SIZE + 1 = 32 + 1 = 33

If we use the dataset:
    - KoNViD_1k
    - Youtube-NT_MovieTrailer
    - Youtube-NT_NationalGeography
    - YUV_4K
    - new_CLIC_20_P_Frames/raw_data, without Animation, Gaming and VR

We can have up to (15 000 000 / NB_FRAMES) different GOP formed with blocks of size
256x256 with no (spatial) overlapping that is:

    Example 0:
        Block[0:256, 0:256] (frame0)
        Block[0:256, 0:256] (frame1)
                - - -
        Block[0:256, 0:256] (frame32)

    Example 1:
        Block[256:512, 256:512] (frame0)
        Block[256:512, 256:512] (frame1)
                - - -
        Block[256:512, 256:512] (frame32)

    Example 3:
        Block[0:256, 0:256] (frame33)
        Block[0:256, 0:256] (frame4)
                - - -
        Block[0:256, 0:256] (frame65)

* 2. Temporal frequency of GOP extraction

Extracting one GOP from frame 0 to NB_FRAMES - 1 (included, so NB_FRAMES frames) 
and then from frame 1 to NB_FRAMES is not worth it memory wise. Thus we will:
    - Set i = 0
    - Extract all non-overlaping (spatially) GOP created from frame
      i to i + NB_FRAMES - 1 (included, so NB_FRAMES frames)
    - Set i = i + NB_FRAMES
meaning that we'll extract GOP every NB_FRAMES frames of the video. No overlapping at all.


* 3. Spatial frequency of GOP extraction

GOP are extracted spatially with no overlaping i.e. 0 to 255, 256 to 511 etc.

* 4. Processing pipeline

The comprehensive pipeline is:

    1. Any video format (.mp4)
    2. Raw yuv video (.yuv)
    3. 3F PNGs for F frames (.png)
    4. Blocks extraction

All steps of the processing pipeline are lossless. For some datasets,
the entire processing pipeline is not needed (we start with .yuv or .png).
In this case, the unnecessary steps are skipped.

* 5. Storing one example

An example is composed of N frames, each of them made of 3 PNGs. In total, 
each example has 3N PNGs. An example is stored as N pickle-serialized 
dictionnaries, one for each frame. Each frame dictionnary containes 3 entries,
'y', 'u', and 'v' and each entry maps to a PIL Image mode 'L'.

* 6. Storing all examples

The 123 456-th will be stored in the folder 1234 under the name 56_<idx_frame>.pkl
They are sorted by <number_of_hundreds>/<remaining>_<idx_frame>.pkl.
Where <idx_frame> goes from 0 to NB_FRAMES - 1 (included)

* 7. Parameters

All parameters related to data extraction and training set building are
defined in the config_file.cfg.
"""


# Everything will be outputed to:
#   OUT_ROOT_PATH = /opt/GPU/Exp/hkzv2358/workspace/GOP_Coding/data_extraction/out/
#
# Inside, we have the training examples stored as .pkl in
#   OUT_TRAINING_SET = OUT_ROOT_PATH + 'training_set/'
#
# All the intermediate files generated from one video (yuv, pgm, png...) are located in
#   TMP_VIDEO_OUT = OUT_ROOT_PATH +  <video_name_wo_ext>/
# Once the crops are extracted, we delete TMP_VIDEO_OUT
#
# We also have the logging file which stores which videos have already been
# processed.
#   OUT_LOGGING_FILE = OUT_ROOT_PATH + 'processed_video.log'
# In this logging file, we store the absolute path of each video.
# Therefore, it should be unique and would allow to restart the extraction
# should it crash.


import configparser
import os
import glob
import argparse
import sys

from utils import get_subdir_in_dir, get_files_in_dir

parser = argparse.ArgumentParser()
parser.add_argument(
    '--quiet',
    help='When used, nothing is printed',
    action='store_true'
)

args = parser.parse_args()

# ================ RETRIEVE VARIABLES FROM CONFIGURATION FILE ================ #
CONFIG_PATH = '/opt/GPU/Exp/hkzv2358/workspace/GOP_Coding/data_extraction/scripts/config_file.cfg'
CONFIG = 'config1'
cfg = configparser.ConfigParser()
cfg.read(CONFIG_PATH)

dataset_list = cfg.get(CONFIG, 'dataset_list').split(',')
ROOT_PATH_DATASET = '/opt/GPU/Dataset/VideoCoding/'
for i in range(len(dataset_list)):
    dataset_list[i] = ROOT_PATH_DATASET + dataset_list[i]
    if not(dataset_list[i].endswith('/')):
        dataset_list[i] += '/'

# ================ RETRIEVE VARIABLES FROM CONFIGURATION FILE ================ #

LJUST_SIZE = 40

# =================== DEFINE THE OUTPUT PATH OF THE PROGRAM ================== #
OUT_ROOT_PATH = '/opt/GPU/Exp/hkzv2358/workspace/GOP_Coding/data_extraction/out/'
OUT_TRAINING_SET = OUT_ROOT_PATH + 'training_set/'
OUT_LOGGING_FILE = OUT_ROOT_PATH + 'processed_video.log'

# Create output folders if needed
os.system('mkdir -p ' + OUT_ROOT_PATH + ' ' + OUT_TRAINING_SET)
# =================== DEFINE THE OUTPUT PATH OF THE PROGRAM ================== #

if not(args.quiet):
    print('[STATE] Start processing: main data extraction script')

# ================= PATH FOR THE DIFFERENT CONVERSION SCRIPTS ================ #
mp4_to_yuv_script = '/opt/GPU/Exp/hkzv2358/workspace/GOP_Coding/data_extraction/scripts/mp4_to_yuv.py'
yuv_to_png_script = '/opt/GPU/Exp/hkzv2358/workspace/GOP_Coding/data_extraction/scripts/yuv_to_png.py'
yuv_to_training_ex_script = '/opt/GPU/Exp/hkzv2358/workspace/GOP_Coding/data_extraction/scripts/png_to_training_ex.py'
# ================= PATH FOR THE DIFFERENT CONVERSION SCRIPTS ================ #

# read, write mode and append at the beginning: use mode r+
# However it can't create the file, so create it if it doesn't exist
if not(os.path.isfile(OUT_LOGGING_FILE)):
    os.system('touch ' + OUT_LOGGING_FILE)

# Check if some examples are already present
progress_log_file = open(OUT_LOGGING_FILE, 'r+')
# After this, the file pointer is at the end
already_processed_files = progress_log_file.read().splitlines()

if (len(already_processed_files) != 0) or (len(get_subdir_in_dir(OUT_TRAINING_SET)) != 0):
    print('[WARNING] Some examples have already been extracted.')
    print('          Use CTRL+C to cancel, and delete files using')
    print('             rm -r  /opt/GPU/Exp/hkzv2358/workspace/GOP_Coding/data_extraction/out/* ')
    # input('          or press ENTER to continue')

progress_log_file.close()

for cur_dataset in dataset_list:
    # "Normal" way, there are only videos inside the dataset folder.
    #
    # For YUV_4K dataset, this is true BUT we don't need to perform the
    # mp4 -> YUV conversion, instead we just have to copy the YUV file
    # already present in the dataset.
    #
    # For CLIC20_P_frame dataset, we already have the PNG, we just have to
    # change their name to <idx_frame>_<channel>.png and we can skip 
    # the mp4 -> yuv -> png conversion.

    if cur_dataset == '/opt/GPU/Dataset/VideoCoding/YUV_4K/':
        flag_yuv_4k_dataset = True
    else:
        flag_yuv_4k_dataset = False

    if cur_dataset == '/opt/GPU/Dataset/VideoCoding/new_CLIC_20_P_Frames/raw_data/':
        flag_clic_p_dataset = True
    else:
        flag_clic_p_dataset = False

    # For the CLIC20 P dataset, videos are not stored as file but as folder
    # with the PNGs frames inside
    if flag_clic_p_dataset:
        list_video_in_dataset = get_subdir_in_dir(cur_dataset)
    else:
        list_video_in_dataset = get_files_in_dir(cur_dataset)
    
    for cur_video in list_video_in_dataset:
        progress_log_file = open(OUT_LOGGING_FILE, 'r+')
        # After this, the file pointer is at the end
        already_processed_files = progress_log_file.read().splitlines()

        # in_file is the absolute path of the video we're processing 
        #
        # As the get subdir and get file functions don't behave the same:
        #   get_subdir_in_dir: return a list of absolute path of all subdirs
        #   get_files_in_dir: return a list of the name of all files in dir
        # So when using the get_files_in_dir function we need to add the root
        # path to get the absolute path

        if flag_clic_p_dataset:
            in_file = cur_video
        else:
            in_file = cur_dataset + cur_video

        if not(args.quiet):
            msg = '\tCurrent video: '.ljust(LJUST_SIZE) + str(in_file).ljust(80)
            print(msg, end='\r', flush=True)
            print('')

        if in_file in already_processed_files:
            if not(args.quiet):
                msg = '\tCurrent video: '.ljust(LJUST_SIZE) + str(in_file).ljust(80) + ' already processed'
                print(msg, end='\r', flush=True)
                print('')
            continue

        name_video_wo_ext = in_file.split('/')[-1].split('.')[0]
        TMP_VIDEO_OUT = OUT_ROOT_PATH +  name_video_wo_ext + '/'

        # 0. Create output directory for this video
        os.system('mkdir -p ' + TMP_VIDEO_OUT)

        # No need to perform the MP4 -> YUV step with the YUV_4K dataset
        if flag_yuv_4k_dataset:
            # Change the name of the video from xxx_hxw_fps_420.yuv to converted_hxw_fps_420
            # and copy it inside TMP_VIDEO_OUT
            yuv_file = TMP_VIDEO_OUT + 'converted_' + '_'.join(in_file.split('/')[-1].split('_')[1:])
            cmd = 'cp ' + in_file + ' ' + yuv_file
            os.system(cmd)
        # Nothing to do for the clic dataset, we already have the PNG files!
        elif flag_clic_p_dataset:
            pass
        else:
            # ================================ MP4 -> YUV ================================ #
            # This will output a file named 
            #   converted_wxh_fps_420.yuv 
            # in the output directory. The name <converted> is fixed, but there should
            # be one output directory for each mp4 so only one converted.yuv file
            cmd = 'python ' + mp4_to_yuv_script + ' -i ' + in_file + ' -o ' + TMP_VIDEO_OUT
            os.system(cmd)

            # yuv_file should be a list of one element containing the absolute path
            # of the converted yuv file.
            yuv_file = glob.glob(TMP_VIDEO_OUT + 'converted_*.yuv')
            if len(yuv_file) > 1:
                print('[ERROR] More than one file matching the pattern <converted_*.yuv>')
                print('        in the output directory.')
                print('\tOutput directory: '.ljust(LJUST_SIZE) + TMP_VIDEO_OUT)
                for i in range(len(yuv_file)):
                    print(('\tFile ' + str(i)).ljust(LJUST_SIZE) + yuv_file[i])

            elif len(yuv_file) == 0:
                print('[ERROR] No file matching the pattern <converted_*.yuv>')
                print('        in the output directory.')
                print('\tOutput directory: '.ljust(LJUST_SIZE) + TMP_VIDEO_OUT)

            yuv_file = yuv_file[0]
            # ================================ MP4 -> YUV ================================ #

        # We just have to move and rename the files inside each folder 
        if flag_clic_p_dataset:
            nb_frame_in_video = int(len(get_files_in_dir(in_file)) / 3)

            # Copy and rename all frames
            # ! Frame counter start at 1 in the CLIC dataset
            for idx_frame in range(nb_frame_in_video):
                for cur_channel in ['y', 'u', 'v']:
                    # Src name = folder with all the png (in file) + name video wo extension + frame + channel
                    src_name = in_file + '/' + name_video_wo_ext + '_' + str(idx_frame + 1).zfill(5) + '_' + cur_channel + '.png'
                    dst_name = TMP_VIDEO_OUT + str(idx_frame) + '_' + cur_channel + '.png'
                    os.system('cp ' + src_name + ' ' + dst_name)

        else:
            # ================================ YUV -> PNG ================================ #
            cmd = 'python ' + yuv_to_png_script + ' -i ' + yuv_file + ' -o ' + TMP_VIDEO_OUT
            os.system(cmd)

            # From now on we don't need the yuv file anymore
            os.system('rm ' + yuv_file)
            # ================================ YUV -> PNG ================================ #

        # =============================== YUV -> Blocks ============================== #
        cmd = 'python ' + yuv_to_training_ex_script
        cmd += ' -i ' + TMP_VIDEO_OUT + ' -o ' + OUT_TRAINING_SET 
        cmd += ' --config ' + CONFIG
        os.system(cmd)

        # From now on we don't need the PNG files or the folder anymore
        os.system('rm -r ' + TMP_VIDEO_OUT + '*.png')
        os.system('rmdir ' + TMP_VIDEO_OUT)
        # =============================== YUV -> Blocks ============================== #

        # Once done, log the video name in the progress_log_file
        progress_log_file.write(in_file + '\n')
        progress_log_file.close()

if not(args.quiet):
    print('[STATE] End processing: main data extraction script')
