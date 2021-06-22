import torch
import torch.nn as nn
from torch.distributions import Normal, Laplace
from torch import log2, clamp, sqrt, zeros_like
from func_util.math_func import PROBA_MIN, PROBA_MAX
from func_util.console_display import print_log_msg


class EntropyCoder(nn.Module):
    """
    Directly estimates the rate from probability
    """
    def __init__(self):
        super(EntropyCoder, self).__init__()

    def forward(self, prob_x, x):
        # Avoid NaN and p_y_tilde > 1
        prob_x = torch.clamp(prob_x, PROBA_MIN, PROBA_MAX)
        rate = -torch.log2(prob_x)

        return rate
