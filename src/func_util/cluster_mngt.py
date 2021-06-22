import os
import torch
import numpy
import random

# Gather in this dic all computation parameters
COMPUTE_PARAM = {
    # In True, print_log_msg doesn't print anything
    'flag_quiet': False,
    # Either 'cpu' or 'cuda:0'
    'device': 'cpu',
    # Wether we're working on cpu or gpu
    'flag_gpu': False,
}


def set_compute_param(key, value):
    COMPUTE_PARAM[key] = value

def seed_all():
    seed = 666

    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_deterministic(True)