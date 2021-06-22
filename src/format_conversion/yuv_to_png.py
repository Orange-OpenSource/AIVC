"""
Convert ONE video file to a sequences of PNG in YUV 420.
"""

import os
import subprocess
import filecmp

from format_conversion.png_to_yuv import png_to_yuv
from format_conversion.utils import get_video_duration, get_video_fps, get_png_resolution, get_video_resolution

def yuv_to_png(in_file, out_dir, start_frame, end_frame, quiet=False, check_lossless=False):
    """
    Convert ONE video file to a sequences of PNG triplets in YUV 420.
    
    in_file: Absolute path of the yuv video file to be processed
    out_dir: Absolute path of the output directory
    start_frame: First frame of the .yuv to be converted
    end_frame: Last frame of the .yuv to be converted. If -1: process until the end
    quiet  : When used, nothing is printed
    check_lossless: when used, the reverse conversion is performed to check
        whether the process is lossless or not

    """

    # Path of the current module, used to obtain the absolute path of the .sh scripts
    current_module_path = os.path.dirname(os.path.realpath(__file__)) + '/'

    LJUST_SIZE = 40

    if not(in_file.endswith('.yuv')):
        print('[ERROR] Input file is not an YUV file')
        print('\tInput file: '.ljust(LJUST_SIZE) + in_file)

    if not(out_dir.endswith('/')):
        out_dir += '/'

    # Real output directory is out_dir/<video_name>/
    os.system('mkdir -p ' + out_dir)

    # Compute number of frames in the YUV ===> Convert all the frames
    if end_frame == -1:
        # - 1 because if we have 30 frame (i.e. 1 second at 30 fps), we have frame from
        # 0 to 29 included
        end_frame = int(get_video_duration(in_file) * get_video_fps(in_file)) - 1

    nb_frame = end_frame - start_frame + 1

    if not(quiet):
        print('[STATE] Start processing: yuv -> png')
        print('\tInput file: '.ljust(LJUST_SIZE) + in_file)
        print('\tOutput directory: '.ljust(LJUST_SIZE) + out_dir)
        print('\tNumber of frames in the video: '.ljust(LJUST_SIZE) + str(nb_frame))

    # ================================ YUV -> PNG ================================ #
    for progress_cnt, i in enumerate(range(start_frame, end_frame + 1)):
        if not(quiet):
            msg = '\tFrame: '.ljust(LJUST_SIZE) + str(progress_cnt + 1).ljust(4) + '/ ' + str(nb_frame)
            print(msg, end='\r', flush=True)

        cmd = current_module_path + 'script_convert_one_frame/yuv_to_png.sh'
        cmd += ' ' + in_file + ' ' + out_dir + ' ' + str(i) + ' '
        cmd += current_module_path + 'script_convert_one_frame/convert_img.py'

        subprocess.call(cmd, shell=True)

    if not(quiet):
        print('')

    if not(quiet):
        print('[STATE] End processing: yuv -> png')
    # ================================ YUV -> PNG ================================ #

    # ================================ PNG -> YUV ================================ #
    if check_lossless:
        if not(quiet):
            print('[STATE] Verifying wether the yuv -> png conversion is lossless')

        # Get the name of the video
        video_name = in_file.split('/')[-1]
        check_yuv_name = out_dir + 'check_' + video_name

        # Convert back the PNG to YUV of name check_yuv_name
        png_to_yuv(out_dir, check_yuv_name, start_frame, end_frame, quiet=quiet)
        
        gnd_truth_name = in_file

        # ===== ONLY IF WE'RE NOT CONVERTING THE ENTIRE YUV VIDEO ===== #    
        # If we don't generate all the frames, this allows to compare that the N
        # first frame are bit exact
        w, h = get_video_resolution(in_file)
        # Number of bytes for the first N frames
        nb_bytes = int(h * w * 1.5)
        gnd_truth_name = out_dir + 'frame' + str(start_frame) + 'to' + str(end_frame) + '_' + video_name
        os.system('rm ' + gnd_truth_name)

        cmd = 'dd skip=' + str(start_frame) + ' count=' + str(end_frame - start_frame + 1) 
        cmd += ' if=' + in_file 
        cmd += ' of=' + gnd_truth_name
        cmd += ' bs=' + str(nb_bytes)
        cmd += ' > /dev/null 2>&1'
        subprocess.call(cmd, shell=True)   
        # ===== ONLY IF WE'RE NOT CONVERTING THE ENTIRE YUV VIDEO ===== #    

        if not(quiet):
            print('')

        flag_lossless = filecmp.cmp(gnd_truth_name, check_yuv_name)
        # Remove the YUV converted back for verification
        os.system('rm ' + check_yuv_name)
        # We have generated a new ground truth with dd, remove!
        if gnd_truth_name != in_file:
            os.system('rm ' + gnd_truth_name)



        print('\tLossless conversion: '.ljust(LJUST_SIZE) + str(flag_lossless))

    # ================================ PNG -> YUV ================================ #

    if not(quiet):
        print('[STATE] End processing')

