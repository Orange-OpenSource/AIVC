# Software Name: AIVC
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

import torch
import numpy
import random
from func_util.console_display import print_log_msg

# Gather in this dic all computation parameters
COMPUTE_PARAM = {
    # Either 'cpu' or 'cuda:0'
    'device': 'cpu',
    # Wether we're working on cpu or gpu
    'flag_gpu': False,
}


def set_compute_param(key, value):
    COMPUTE_PARAM[key] = value

def seed_all():
    seed = 666

    print_log_msg('INFO', 'Seed', '', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_deterministic(True)