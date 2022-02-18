# Software Name: AIVC
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

"""
Define different gop structure in this file.

A GOP is a dictionnary of dictionnaries. First entries are called
<frame_i> where i is the index of the DISPLAY order (i.e. the temporal index).

Then for each frame we have:
    - The type (either i, p or b).
    - The name of the frame to used for previous or next reference.
    - The coding order: 0 is coded first, then 1, then 2...
"""
# Each frame has an ID to identify its type
FRAME_I = 0
FRAME_P = 1
FRAME_B = 2

# ======= GENERATE RANDOM ACCESS GOPS WITH A RECURSIVE FUNCTION ======= #
def generate_ra_gop(gop_size):
    gop = {}

    gop['frame_0'] = {
        'type': FRAME_I,
        'prev_ref': None,
        'next_ref': None,
        'coding_order': 0,
    }

    gop['frame_' + str(gop_size)] = {
        'type': FRAME_P,
        'prev_ref': 'frame_0',
        'next_ref': None,
        'coding_order': 1,
    }

    N = int(gop_size / 2)
    cnt_coding = 2
    frame_idx = N
    gop, _ = do_next_temp_layer(frame_idx, N, cnt_coding, gop)

    return gop


def do_next_temp_layer(frame_idx, cur_N, cnt_coding, gop):

    gop['frame_' + str(frame_idx)] = {
        'type': FRAME_B,
        'prev_ref': 'frame_' + str(frame_idx - cur_N),
        'next_ref': 'frame_' + str(frame_idx + cur_N),
        'coding_order': cnt_coding,
    }
    cnt_coding += 1
    cur_N = int(cur_N / 2)

    if cur_N != 0:
        gop, cnt_coding = do_next_temp_layer(frame_idx - cur_N, cur_N, cnt_coding, gop)
        gop, cnt_coding = do_next_temp_layer(frame_idx + cur_N, cur_N, cnt_coding, gop)

    return gop, cnt_coding


def chained_gop(gop_size, n):
    """
    Create a gop structure of <n> chained GOP. There is a single I for the
    whole gop structure.
    """
    first_gop = generate_ra_gop(gop_size)

    for i in range(1, n):
        cur_gop = generate_ra_gop(gop_size)
        # Remove the I frame of the gop we want to chain
        cur_gop.pop('frame_0')
        # We need to change all the frame indices of cur_gop
        # Example, gop 4:
        # first_gop = 0, 1, 2, 3, 4
        # cur_gop   = 1, 2, 3, 4
        # we want the frame names in cur gop to be 5, 6, 7, 8 so we add
        # an offset of i * gop_size
        frame_idx_offset = i * gop_size
        for frame_name in cur_gop:
            old_frame_idx = int(frame_name.split('_')[-1])
            new_frame_idx = old_frame_idx + frame_idx_offset

            # Append the shifted frame to the final gop structure
            new_frame_name = 'frame_' + str(new_frame_idx)
            first_gop[new_frame_name] = cur_gop.get(frame_name)

            # Also shift the references
            old_prev_ref = first_gop.get(new_frame_name).get('prev_ref')
            old_next_ref = first_gop.get(new_frame_name).get('next_ref')
            # It is none for the p_frame
            if not(old_next_ref is None):
                new_next_ref = 'frame_' + str(int(old_next_ref.split('_')[-1]) + frame_idx_offset)
            else:
                new_next_ref = old_next_ref

            new_prev_ref = 'frame_' + str(int(old_prev_ref.split('_')[-1]) + frame_idx_offset)
            first_gop[new_frame_name]['prev_ref'] = new_prev_ref
            first_gop[new_frame_name]['next_ref'] = new_next_ref

            # And shift the coding order
            first_gop[new_frame_name]['coding_order'] += frame_idx_offset

    return first_gop


def generate_ldp_gop(gop_size):
    """
    Generate a low-delay P gop of one I + <gop_size> P-framess
    """

    gop = {}
    gop['frame_0'] = {
        'type': FRAME_I,
        'prev_ref': None,
        'next_ref': None,
        'coding_order': 0,
    }

    # For i in [1, gop_size] included!
    for i in range(1, gop_size + 1):
        gop['frame_' + str(i)] = {
            'type': FRAME_P,
            'prev_ref': 'frame_' + str(i - 1),
            'next_ref': None,
            'coding_order': i,
        }

    return gop
# ======= GENERATE RANDOM ACCESS GOPS WITH A RECURSIVE FUNCTION ======= #


# =============== USEFUL FUNCTIONS FOR GOP MANAGEMENT ================= #
def get_gop_struct_name(GOP_struct):
    for k in GOP_STRUCT_DIC:
        if GOP_STRUCT_DIC.get(k) == GOP_struct:
            return k


def get_name_frame_code(GOP_struct, idx_code):
    """
    Return a list of name (frame_i) with all frames that have
    coding_order = <idx_code> in the given GOP structure <GOP_struct>.
    """

    name_frame_code = []

    for f in GOP_struct:
        if GOP_struct.get(f).get('coding_order') == idx_code:
            name_frame_code.append(f)

    # If nothing has been found, return the empty list
    return name_frame_code


def get_depth_gop(GOP_struct):
    """
    Return the maximal depth (i.e. the maximal coding order) of a GOP
    """

    depth_max = 0
    for f in GOP_struct:
        cur_coding_order = GOP_struct.get(f).get('coding_order')
        if cur_coding_order > depth_max:
            depth_max = cur_coding_order
    return depth_max


def get_current_gop_struct(nb_epoch_done, change_gop_struct_epoch, training_gop_struct):
    """
    According to the number of epoch done, look at change_gop_struct epoch to check
    which stage we're doing. Then return the correct stage, selected in
    training_gop_struct
    """

    for i in range(len(change_gop_struct_epoch)):
        # Epoch where the previous training stage ended, 0 if
        # we're still doing the first stage
        end_prev_stage = 0 if i == 0 else change_gop_struct_epoch[i - 1]
        # Epoch where the next stage begins
        start_next_stage = change_gop_struct_epoch[i]

        # We have done the previous stage but we're not yet to the next one
        if (nb_epoch_done < start_next_stage) and (nb_epoch_done >= end_prev_stage):
            cur_gop_struct = training_gop_struct[i]

    return cur_gop_struct
# =============== USEFUL FUNCTIONS FOR GOP MANAGEMENT ================= #


def generate_gop_struct(gop_struct_name: str):

    # All intra configurations
    if gop_struct_name == '1_GOP_0':
        gop_struct = {
            'frame_0': {
                'type': FRAME_I,
                'prev_ref': None,
                'next_ref': None,
                'coding_order': 0,
            },
        }
    # Low delay P configuration
    elif 'LDP' in gop_struct_name.split('_'):
        gop_size = int(gop_struct_name.split('_')[-1])
        gop_struct = generate_ldp_gop(gop_size)
    # Random access configuration
    else:
        n_chained_gops = int(gop_struct_name.split('_')[0])
        gop_size = int(gop_struct_name.split('_')[-1])
        gop_struct = chained_gop(gop_size, n_chained_gops)

    return gop_struct

# # ALL INTRA GOP
# GOP_0 = {
#     'frame_0': {
#         'type': FRAME_I,
#         'prev_ref': None,
#         'next_ref': None,
#         'coding_order': 0,
#     },
# }

# GOP_STRUCT_DIC = {'1_GOP_0': GOP_0}
# # Generate RA gop
# for gop_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
#     for n_gops in range(1, 512):
#         k = f'{n_gops}_GOP_{gop_size}'
#         GOP_STRUCT_DIC[k] = chained_gop(gop_size, n_gops)

# # Generate LDP gop:
# for gop_size in range(2, 1024):
#     k = f'LDP_{gop_size}'
#     GOP_STRUCT_DIC[k] = generate_ldp_gop(gop_size)

# print(len(GOP_STRUCT_DIC))


