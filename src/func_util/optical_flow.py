import torch
import torch.nn as nn
from torch.autograd import Variable

from func_util.nn_util import get_value

def warp(x, flo, interpol_mode='bilinear', padding_mode='border', align_corners=True):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    cur_device = x.device

    # mesh grid 
    xx = torch.arange(0, W, device=cur_device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=cur_device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    # Already on cur_device as xx and yy are created on cur_device
    grid = torch.cat((xx, yy), 1).float()

    vgrid = Variable(grid) + flo

    # scale grid to [-1, 1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)


    output = nn.functional.grid_sample(
        x, vgrid, mode=interpol_mode, padding_mode=padding_mode, align_corners=align_corners
    )
    mask = torch.autograd.Variable(torch.ones(x.size(), device=cur_device))
    mask = nn.functional.grid_sample(
        mask, vgrid, mode=interpol_mode, padding_mode=padding_mode, align_corners=align_corners
    )

    # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask