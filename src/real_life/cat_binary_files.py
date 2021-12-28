# Software Name: AIVC
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

import os
import glob

from func_util.nn_util import get_value
from func_util.console_display import print_log_msg
from real_life.utils import GOP_SUFFIX, GOP_HEADER_SUFFIX, BITSTREAM_SUFFIX, VIDEO_HEADER_SUFFIX
from real_life.header import write_video_header


def cat_one_gop(param):

    """
    This function is called at the end of GOP_forward. Its role is to cat the
    bitstream for the entire GOP into a single file.

    Use <idx_gop> to get the GOP header named <idx_gop><GOP_HEADER_SUFFIX>. Then, look for all files
    without <GOP_HEADER_SUFFIX> and <GOP_SUFFIX> in their name. This is all the frames
    of the GOP. They are concatenated into a single <idx_gop><GOP_SUFFIX>.

    The structure of the resulting file is as follows:

    [GOP_Header       (X bytes)]

    [Size of frame 0  (4 bytes)]
    [Frame 0         (variable)]

    [Size of frame 1  (4 bytes)]
    [Frame 1         (variable)]

    [Size of frame N  (4 bytes)]
    [Frame N         (variable)]

    """

    DEFAULT_PARAM = {
        # Idx of the GOP in the sequence.
        'idx_gop': 0,
        # Directory where all the bitstream files are located.
        'bitstream_dir': '',
    }

    # =========================== RETRIEVE INPUTS =========================== #
    idx_gop = get_value('idx_gop', param, DEFAULT_PARAM)
    bitstream_dir = get_value('bitstream_dir', param, DEFAULT_PARAM)
    # =========================== RETRIEVE INPUTS =========================== #

    if not(bitstream_dir.endswith('/')):
        bitstream_dir += '/'

    gop_header = bitstream_dir + str(idx_gop) + GOP_HEADER_SUFFIX

    # Get the index of all the compressed frames
    # This loop is for getting the smallest frame index
    # No need to sort the glob.glob
    list_all_files = glob.glob(bitstream_dir + '*')
    list_idx_frame = []
    for file_name in list_all_files:
        # We only want to process frame file, not gop header, nor already concatenated gop
        # nor the data_dim.pkl file
        if not(file_name.endswith(GOP_HEADER_SUFFIX)) and not(file_name.endswith(GOP_SUFFIX)) and\
           not(file_name.endswith('data_dim.pkl')):
            list_idx_frame.append(
                int(file_name.rstrip(BITSTREAM_SUFFIX).split('/')[-1])
            )

    first_GOP_frame_idx = min(list_idx_frame)

    # Into the output file
    out_file = bitstream_dir + str(idx_gop) + GOP_SUFFIX
    byte_to_write = b''

    with open(gop_header, 'rb') as fin:
        byte_to_write += fin.read()

    # Remove gop header separate file
    os.system('rm ' + gop_header)

    # Concatenate all the files
    for i in range(first_GOP_frame_idx, first_GOP_frame_idx + len(list_idx_frame)):
        frame_name = bitstream_dir + str(i) + BITSTREAM_SUFFIX
        size_frame = os.path.getsize(frame_name)

        with open(frame_name, 'rb') as fin:
            byte_to_write += int(size_frame).to_bytes(4, byteorder='big')
            byte_to_write += fin.read()

        # Remove this frame separate file
        os.system('rm ' + frame_name)

    # Write to the output file
    with open(out_file, 'wb') as fout:
        fout.write(byte_to_write)


def cat_one_video(param):
    """
    This function is called at the end of infer_one_sequence. Its role is to
    cat the bitstream for the entire sequences into a single file.

    Looking in a bitstream directory, it takes all the GOP files
    <idx_GOP><GOP_SUFFIX> and concatenate them into a single file named
    <bitstream_dir>.bin, stored in the root RealLife/data/

    In the end we remove bitstream_dir

    The structure of the resulting file is as follows:

    [Video header     (X bytes)]

    [Size of GOP 0    (4 bytes)]
    [GOP 0           (variable)]

    [Size of GOP 1    (4 bytes)]
    [GOP 1           (variable)]

    [Size of GOP N    (4 bytes)]
    [GOP N           (variable)]
    """

    DEFAULT_PARAM = {
        # Directory where all the bitstream files are located.
        'bitstream_dir': '',
        # Index of the first frame to code
        'idx_starting_frame': 1,
        # Index of the last frame to code (included)
        'idx_end_frame': None,
        # Path of the final bitstream file
        'final_bitstream_path': '',
    }

    # =========================== RETRIEVE INPUTS =========================== #
    bitstream_dir = get_value('bitstream_dir', param, DEFAULT_PARAM)
    idx_starting_frame = get_value('idx_starting_frame', param, DEFAULT_PARAM)
    idx_end_frame = get_value('idx_end_frame', param, DEFAULT_PARAM)
    final_bitstream_path = get_value('final_bitstream_path', param, DEFAULT_PARAM)
    # =========================== RETRIEVE INPUTS =========================== #

    if not(bitstream_dir.endswith('/')):
        bitstream_dir += '/'

    list_gop_files = sorted(glob.glob(bitstream_dir + '*' + GOP_SUFFIX))

    # Temporary video header file: we write it, then we take its raw bytes to
    # concatenate them to the rest of the video bitstream. At the end, delete
    # the video header file
    video_header_path = bitstream_dir + VIDEO_HEADER_SUFFIX
    write_video_header({
        'nb_gop': len(list_gop_files),
        'header_path': video_header_path,
        'idx_starting_frame': idx_starting_frame,
        'idx_end_frame': idx_end_frame,
    })
    byte_to_write = b''
    with open(video_header_path, 'rb') as fin:
        byte_to_write += fin.read()
    os.system('rm ' + video_header_path)

    # Loop on all the gop files
    for idx_gop in range(len(list_gop_files)):
        f = bitstream_dir + str(idx_gop) + GOP_SUFFIX
        # print_log_msg('DEBUG', 'cat_one_video', 'gop_file', f)
        if not(f.endswith(GOP_SUFFIX)):
            print('[ERROR] cat_one_video should only process gop_files')
            print('[ERROR] current file is: ' + str(f))
            continue

        # Append size of the gop file + its content to byte_to_write
        size_file = os.path.getsize(f)
        with open(f, 'rb') as fin:
            byte_to_write += int(size_file).to_bytes(4, byteorder='big')
            byte_to_write += fin.read()

        # Delete the gop file
        os.system('rm ' + f)

    # Finally, write all the video bytes to a single out file
    out_file = final_bitstream_path

    # Create the parent directory of out_file if needed
    parent_dir_path = '/'.join(out_file.rstrip('/').split('/')[:-1])
    os.system('mkdir -p ' + parent_dir_path)

    with open(out_file, 'wb') as fout:
        fout.write(byte_to_write)

    # # Remove the bitstream dir (it should be empty)
    # os.system('rmdir ' + bitstream_dir)
    # # Remove the .bin at the end of the out file and we're done!
    # os.system('mv ' + out_file + ' ' + out_file.rstrip('.bin'))
