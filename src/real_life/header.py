# Software Name: AIVC
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

import os
import math
import pickle # Needed to save the temporary data_dim dictionnary

from func_util.nn_util import get_value
from func_util.GOP_structure import GOP_STRUCT_DIC
from real_life.utils import BITSTREAM_SUFFIX, GOP_HEADER_SUFFIX, VIDEO_HEADER_SUFFIX

"""
The bitstream needs to embeds some information such as the data
size encoded-decoded by arithmetic coding.

GOP header format:
------------------

[idx_gop_struct (1 byte)]
[idx_rate       (1 byte)] (can be decimal but will be multiplied by 16 and stored on one byte)

Video header format:
--------------------

(x, y and z referes to the input data, the latents and the hyperprior)

[H_x (2 bytes)] [W_x (2 bytes)]
[H_y (2 bytes)] [W_y (2 bytes)]
[H_z (2 bytes)] [W_z (2 bytes)]
[Nb GOP              (2 bytes)]
[Index first frame   (2 bytes)]  Code from idx_first to idx_last included
[Index last frame    (2 bytes)]  This is needed to remove the padded frames
"""

# Used in the header:
GOP_NAME_LIST = list(GOP_STRUCT_DIC.keys())

def write_video_header(param):
    DEFAULT_PARAM = {
        # Path of the header
        'header_path': None,
        # Number of GOPs in the video sequence
        'nb_gop': 0,
        # Index of the first frame to code
        'idx_starting_frame': 1,
        # Index of the last frame to code (included)
        'idx_end_frame': None,
    }

    header_path = get_value('header_path', param, DEFAULT_PARAM)
    nb_gop = get_value('nb_gop', param, DEFAULT_PARAM)
    idx_starting_frame = get_value('idx_starting_frame', param, DEFAULT_PARAM)
    idx_end_frame = get_value('idx_end_frame', param, DEFAULT_PARAM)

    if not(header_path.endswith(VIDEO_HEADER_SUFFIX)):
        header_path += VIDEO_HEADER_SUFFIX

    # Read the data dim pickle dictionnary
    data_dim_path = '/'.join(header_path.split('/')[:-1]) + '/data_dim.pkl'
    # print('data_dim_path: ' + data_dim_path)

    with open(data_dim_path, 'rb') as fin:
        data_dim = pickle.load(fin)
    # Remove the data_dim pickle dictionnary
    os.system('rm ' + data_dim_path)


    H_x, W_x = data_dim.get('x')
    H_y, W_y = data_dim.get('y')
    H_z, W_z = data_dim.get('z')

    byte_to_write = b''
    for tmp in [H_x, W_x, H_y, W_y, H_z, W_z, nb_gop, idx_starting_frame, idx_end_frame]:
        byte_to_write += tmp.to_bytes(2, byteorder='big')

    with open(header_path, 'wb') as fout:
        fout.write(byte_to_write)


def read_video_header(param):

    DEFAULT_PARAM = {
        # Path of the header
        'header_path': None,
    }

    header_path = get_value('header_path', param, DEFAULT_PARAM)

    if not(header_path.endswith(VIDEO_HEADER_SUFFIX)):
        header_path += VIDEO_HEADER_SUFFIX

    with open(header_path, 'rb') as fin:
        byte_stream = fin.read()

    H_x = int.from_bytes(byte_stream[0:2], byteorder='big')
    W_x = int.from_bytes(byte_stream[2:4], byteorder='big')

    H_y = int.from_bytes(byte_stream[4:6], byteorder='big')
    W_y = int.from_bytes(byte_stream[6:8], byteorder='big')

    H_z = int.from_bytes(byte_stream[8:10], byteorder='big')
    W_z = int.from_bytes(byte_stream[10:12], byteorder='big')

    nb_gop = int.from_bytes(byte_stream[12:14], byteorder='big')

    idx_starting_frame = int.from_bytes(byte_stream[14:16], byteorder='big')
    idx_end_frame = int.from_bytes(byte_stream[16:18], byteorder='big')


    # x refers to input_data channel Y
    # x_uv refers to input_data channel U
    # ! For now it is rounded through ceil when x is not a multiple of 2
    data_dim = {
        'x': (H_x, W_x),
        'y': (H_y, W_y),
        'z': (H_z, W_z),
        'x_uv': (math.ceil(H_x / 2), math.ceil(W_x / 2)),
    }

    return data_dim, nb_gop, idx_starting_frame, idx_end_frame


def write_gop_header(param):

    DEFAULT_PARAM = {
        # Path of the header
        'header_path': None,
        # Index of the rd-point. It can be a floating point, but it will be
        # multiplied by 16 to be stored as an int (on 1 byte). As such
        # it can only be in [0, 15]
        'idx_rate': 0.,
        # GOP structure name, must be a string!,
        'GOP_struct_name': '',
        # Data dimension (resolution) (as dictionnary of tuples), one for 'x', one for 'y'
        # and one for 'z'. The data dimension dictionnary is directly in the bitstream
        # dir. It is overwritten at each GOP forward (not really elegant...)
        # At the end of the encoding process it is loaded in the
        # write_header_function to be properly written in the binary file.
        'data_dim': None,
    }

    header_path = get_value('header_path', param, DEFAULT_PARAM)
    GOP_struct_name = get_value('GOP_struct_name', param, DEFAULT_PARAM)
    idx_rate = get_value('idx_rate', param, DEFAULT_PARAM)
    data_dim = get_value('data_dim', param, DEFAULT_PARAM)

    if not(header_path.endswith(GOP_HEADER_SUFFIX)):
        header_path += GOP_HEADER_SUFFIX

    byte_to_write = b''

    idx_gop_struct = GOP_NAME_LIST.index(GOP_struct_name)
    byte_to_write += idx_gop_struct.to_bytes(1, byteorder='big')

    casted_idx_rate = int(round(idx_rate * 16))
    byte_to_write += casted_idx_rate.to_bytes(1, byteorder='big')

    with open(header_path, 'wb') as fout:
        fout.write(byte_to_write)

    data_dim_path = '/'.join(header_path.split('/')[:-1]) + '/data_dim.pkl'
    # print('data_dim_path: ' + data_dim_path)
    with open(data_dim_path, 'wb') as fout:
        pickle.dump(data_dim, fout, pickle.HIGHEST_PROTOCOL)


def read_gop_header(param):
    """
    Read the dimension data from the header.
    """

    DEFAULT_PARAM = {
        # Path of the header
        'header_path': None,
    }

    header_path = get_value('header_path', param, DEFAULT_PARAM)

    if not(header_path.endswith(GOP_HEADER_SUFFIX)):
        header_path += GOP_HEADER_SUFFIX

    with open(header_path, 'rb') as fin:
        byte_stream = fin.read()

    idx_gop_struct = byte_stream[0]
    idx_rate = byte_stream[1] / 16

    GOP_struct = GOP_STRUCT_DIC.get(GOP_NAME_LIST[idx_gop_struct])

    return GOP_struct, idx_rate

