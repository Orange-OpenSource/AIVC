import torch
from torch.nn import Module

from func_util.nn_util import get_value, push_dic_to_device
from func_util.img_processing import get_y_u_v
from func_util.ms_ssim import MSSSIM
from func_util.cluster_mngt import COMPUTE_PARAM

class EntropyDistLossEndToEnd(Module):
    """
    Documentation can be found at:
        https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
    """

    def __init__(self, mode='mse'):
        """
        Train a network by minimising both the entropy and the distortion.
        mode is either:
            'mse': minimize mean squared error
        """
        super(EntropyDistLossEndToEnd, self).__init__()

        # ! Mode is deprecated and not used anymore
        self.mode = mode

        # self.mse_fn = MSELoss()
        # self.msssim_fn = MSSSIMLoss()

    def forward(self, param):
        """
        Compute the loss: Distortion + lambda x Rate

        target denotes the target frame
        net_out gathers all the output from the network inside a dic
        """
        
        DEFAULT_PARAM = {
            # The uncompressed frames (i.e. the frames to code), defined as:
            #   frame_0: {'y': tensor, 'u': tensor, 'v': tensor}
            #   frame_1: {'y': tensor, 'u': tensor, 'v': tensor}
            'target': None,
            # The output of the model. Stored as a dic of dict, with one
            # entry (and one dic) for each frame
            'net_out': None,
            # Lamdaa for CodecNet rate
            'l_codec': 0.,
            # Lambda for MOFNet rate
            'l_mof': 0.,
            # Weighting for the I-frame loss
            'weight_i_frame_loss': 1.,
            # Which distortion to we use for the optimization process
            #   - <mse>
            #   - <ms_ssim> (use 1 - ms_ssim as distortion)
            'dist_loss': 'mse',
            # Weighting applied to the entire GOP loss, used for the
            #  multi-rate training.
            'loss_weighting': 1.,
        }

        # ========== RETRIEVE INPUTS ========== #
        target = get_value('target', param, DEFAULT_PARAM)
        net_out = get_value('net_out', param, DEFAULT_PARAM)
        l_codec = get_value('l_codec', param, DEFAULT_PARAM)
        l_mof = get_value('l_mof', param, DEFAULT_PARAM)
        weight_i_frame_loss = get_value('weight_i_frame_loss', param, DEFAULT_PARAM)
        dist_loss = get_value('dist_loss', param, DEFAULT_PARAM)
        loss_weighting = get_value('loss_weighting', param, DEFAULT_PARAM)
        # ========== RETRIEVE INPUTS ========== #

        # Things to display in the tensorboard
        logging_dic = {}

        # Compute loss, and generate a bunch of metric in result
        final_loss, result = compute_metrics_one_GOP({
            'net_out': net_out,
            'target': target,
            'l_mof': l_mof,
            'l_codec': l_codec,
            'weight_i_frame_loss': weight_i_frame_loss,
            'dist_loss': dist_loss,
            'loss_weighting': loss_weighting,
        })

        # Refactor the entries of result to store them in logging dic
        # logging_dic entries are: <value_name>/<frame_name>
        for f in result:
            cur_frame_res = result.get(f)
            for k in cur_frame_res:
                logging_dic[k + '/' + f] = cur_frame_res.get(k)

        return final_loss, logging_dic


def compute_metrics_one_GOP(param):
    """
    From the net_out dictionnary of the GOP_forward function, 
    compute a bunch of metrics for each frames and for the entire
    GOP. The result structure is:
    result = {
      'frame_0': {a: 4, b: 2},
      'frame_1': {a: 0, b: 3},
      'GOP':     {a: 1, b: 8},
    }
    """
    DEFAULT_PARAM = {
        # The net_out dictionnary from the GOP_forward function
        'net_out': None,
        # The target frames i.e. the un-compressed code
        'target': None,
        # Lamdaa for CodecNet rate
        'l_codec': 0.,
        # Lambda for MOFNet rate
        'l_mof': 0.,
        # Weighting for the I-frame loss
        'weight_i_frame_loss': 1.,
        # Which distortion to we use for the optimization process
        #   - <mse>
        #   - <ms_ssim> (use 1 - ms_ssim as distortion)
        'dist_loss': 'mse',
        # Weighting applied to the entire GOP loss, used for the
        #  multi-rate training.
        'loss_weighting': 1.,
        # Number of padded frames, i.e. the last <nb_pad_frame> are not
        # real frames, they are just here to complete a GOP. 
        # When averaging the GOP metrics, we want to take the rate into
        # account, but not the distortion.
        'nb_pad_frame': 0,
    }

    # ========== RETRIEVE INPUTS ========== #
    net_out = get_value('net_out', param, DEFAULT_PARAM)
    target = get_value('target', param, DEFAULT_PARAM)
    l_codec = get_value('l_codec', param, DEFAULT_PARAM)
    l_mof = get_value('l_mof', param, DEFAULT_PARAM)
    weight_i_frame_loss = get_value('weight_i_frame_loss', param, DEFAULT_PARAM)
    dist_loss = get_value('dist_loss', param, DEFAULT_PARAM)
    loss_weighting = get_value('loss_weighting', param, DEFAULT_PARAM)
    nb_pad_frame = get_value('nb_pad_frame', param, DEFAULT_PARAM)
    # ========== RETRIEVE INPUTS ========== #

    cur_device = COMPUTE_PARAM.get('device')
    mse_fn = MSELoss()
    ms_ssim_fn = MSSSIMLoss()

    # The thing to output
    result = {}

    # Name of the different rate tensors to retrieve in net_out
    rate_list = ['mode_rate_y', 'mode_rate_z', 'codec_rate_y', 'codec_rate_z']

    # Rate by frame and by name, for an easier management
    rate_dic_bpp = {}

    # Dimension of the Y plane of one frame:
    B, C, H, W = target.get('frame_0').get('y').size()
    # Number of pixel in a Y frame
    nb_pixel = H * W
    # Loss will be acumulated in this variable after each frame
    final_loss = torch.tensor([0.], device=cur_device)

    # Iterate on all frames
    for f in target:
        # All outputs related to the f frame
        cur_frame_out = net_out.get(f)
        # Initialize current result dic:
        result[f] = {}

        # ======= RETRIEVE STUFF ======== #
        cur_target = target.get(f)
        x_hat = cur_frame_out.get('x_hat')

        cur_alpha = cur_frame_out.get('alpha')
        cur_beta = cur_frame_out.get('beta')

        rate_dic_bpp[f] = {}
        for rate_name in rate_list:
            rate_dic_bpp[f][rate_name] = cur_frame_out.get(rate_name).sum() / (B * nb_pixel)
        # ======= RETRIEVE STUFF ======== #

        # Push to device if needed
        x_hat = push_dic_to_device(x_hat, cur_device)
        cur_target = push_dic_to_device(cur_target, cur_device)

        # ========= DISTORTION ========= #
        mse = mse_fn(x_hat, cur_target)
        ms_ssim_loss, ms_ssim = ms_ssim_fn(x_hat, cur_target)
        
        if dist_loss == 'mse':
            dist = mse
        elif dist_loss == 'ms_ssim':
            dist = ms_ssim_loss

        # These are 444 tensor and not 420 YUV dictionnary
        warping = cur_frame_out.get('warping')
        code = cur_frame_out.get('code')
        mse_warping = ((warping - code) ** 2).mean()
        # ========= DISTORTION ========= #

        # ============ RATE ============ #
        # Add some entries in rate_dic_bpp:
        rate_dic_bpp[f]['mode_rate'] = rate_dic_bpp.get(f).get('mode_rate_y') + rate_dic_bpp.get(f).get('mode_rate_z')
        rate_dic_bpp[f]['codec_rate'] = rate_dic_bpp.get(f).get('codec_rate_y') + rate_dic_bpp.get(f).get('codec_rate_z')
        rate_dic_bpp[f]['total_rate'] = rate_dic_bpp.get(f).get('mode_rate') + rate_dic_bpp.get(f).get('codec_rate')
        # ============ RATE ============ #

        # ======= RATE-DISTORTION ====== #
        cur_loss = l_codec * rate_dic_bpp.get(f).get('codec_rate') +\
                   l_mof * rate_dic_bpp.get(f).get('mode_rate') +\
                   dist

        # Ponderate all frames with loss_weighting + the I-frame with 
        # weight_i_frame_loss
        if f == 'frame_0':
            weighted_loss = cur_loss * weight_i_frame_loss * loss_weighting
        else:
            weighted_loss = cur_loss * loss_weighting

        final_loss += weighted_loss
        # ======= RATE-DISTORTION ====== #
        
        # ========== LOGGING =========== #
        result[f]['loss'] = weighted_loss.detach()
        result[f]['mse'] = mse.detach()
        result[f]['mse_warping'] = mse_warping.detach()
        result[f]['psnr'] = 10 * torch.log10(1. / result.get(f).get('mse'))
        result[f]['psnr_warping'] = 10 * torch.log10(1. / result.get(f).get('mse_warping'))
        result[f]['codec_rate_bpp'] = rate_dic_bpp.get(f).get('codec_rate').detach()
        result[f]['mode_rate_bpp'] = rate_dic_bpp.get(f).get('mode_rate').detach()
        result[f]['total_rate_bpp'] = rate_dic_bpp.get(f).get('total_rate').detach()
        result[f]['mean_alpha'] = cur_alpha.mean()
        result[f]['mean_beta'] = cur_beta.mean()
        result[f]['ms_ssim'] = ms_ssim.detach()
        result[f]['ms_ssim_db'] = -10.0 * torch.log10(ms_ssim_loss)
        result[f]['h'] = torch.tensor([float(H)], device=cur_device)
        result[f]['w'] = torch.tensor([float(W)], device=cur_device)
        # ========== LOGGING =========== #

        # Push back to cpu
        x_hat = push_dic_to_device(x_hat, 'cpu')
        cur_target = push_dic_to_device(cur_target, 'cpu')

    # Previous logging what a frame by frame logging, here we log the final
    # (i.e. on the whole GOP) values
    result['GOP'] = average_N_frame({'x': result, 'nb_pad_frame': nb_pad_frame})

    # We return both final_loss (without detach) for the gradient backprop
    # and result for logging purpose.
    return final_loss, result


def average_N_frame(param):
    """
    From an input dictionnary of dictionnaries structured as
    result = {
      'frame_0': {a: 4, b: 2},
      'frame_1': {a: 0, b: 3},
    }

    compute an average dictionnary which takes the average of all of the 
    N frames present in result.
    """
    DEFAULT_PARAM = {
        # The dictionnary to be averaged 
        'x': None,
        # Number of padded frames, i.e. the last <nb_pad_frame> are not
        # real frames, they are just here to complete a GOP. 
        # When averaging the GOP metrics, we want to take the rate into
        # account, but not the distortion.
        'nb_pad_frame': 0,
    }

    # ========== RETRIEVE INPUTS ========== #
    x = get_value('x', param, DEFAULT_PARAM)
    nb_pad_frame = get_value('nb_pad_frame', param, DEFAULT_PARAM)
    # ========== RETRIEVE INPUTS ========== #

    average = {}
    # Number of frames (normal + padded) in the GOP
    nb_frame = len(x)
    # Number of non-padded frames (i.e. normal frames)
    nb_non_padded = nb_frame - nb_pad_frame
    # Keys that are ignored by padded frames when computing the average
    exclude_padded = ['mse', 'mse_warping', 'ms_ssim', 'ms_ssim_db', 'psnr']

    # Initialize an all zero average dictionnary with the same entry 
    # as in a frame result dictionnary
    for f in x:
        cur_frame = x.get(f)
        for k in cur_frame:
            average[k] = torch.tensor([0.], device=cur_frame.get(k).device)
        # We only need to do it for one frame
        break

    # Accumulate the results from all frame
    for idx_frame, f in enumerate(x):
        cur_frame = x.get(f)
        for k in cur_frame:
            # Check if it is a padded frame.
            # For padded frame, we don't accumulate the distortion
            if (idx_frame >= nb_non_padded) and (k in exclude_padded):
                    continue
            average[k] += cur_frame.get(k)
    
    # Normalize by the number of frames
    # print(nb_frame)

    for f in x:
        cur_frame = x.get(f)
        idx_frame = int(f.split('_')[-1])
        # When averaging, we have two normalization factors. 
        # For distortion, we compute it only on the non-padded
        # frames, thus we divide by nb_non_padded.
        # For everything else we compute it on all frames so we 
        # divide it by nb_frame
        for k in cur_frame:
            if k in exclude_padded:
                average[k] /= nb_non_padded
            else:
                average[k] /= nb_frame
        # We only need to average once!
        break


    # We don't want to have a average of PSNR but an average of MSE
    average['psnr'] = 10 * torch.log10(1. / average.get('mse'))
    average['psnr_warping'] = 10 * torch.log10(1. / average.get('mse_warping'))
    # We don't want to have an average of ms_ssim_db but an average of ms_ssim
    average['ms_ssim_db'] = -10.0 * torch.log10(1 - average.get('ms_ssim'))

    return average


class EntropyMSELoss(Module):
    """
    Documentation can be found at:
        https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
    """

    def __init__(self, mode='mse'):
        super(EntropyMSELoss, self).__init__()

        self.mode = mode

        self.mse_fn = MSELoss()
        self.msssim_fn = MSSSIMLoss()

    def forward(self, x_hat, code, rate_y, rate_z, lambda_tradeoff):
        """
        Compute the loss: Distortion + lambda x Rate
        With Distortion = || x _hat- x_code ||^2
        """
        # ========= DISTORTION ========= #
        mse = self.mse_fn(x_hat, code)
        # loss_ms_ssim = 1 - ms_ssim
        loss_ms_ssim, ms_ssim = self.msssim_fn(x_hat, code)

        if self.mode == 'mse':
            dist = mse
        elif self.mode == 'ms_ssim':
            dist = loss_ms_ssim

        ms_ssim_db = -10.0 * torch.log10(loss_ms_ssim)
        # ========= DISTORTION ========= #

        # ============ RATE ============ #
        # rate shape is
        # [nb_ex_minibatch, nb_features btle, spatial_dim, spatial_dim]

        # Rate per pixel is total rate / number pixel in y
        nb_pixel = x_hat.get('y').size()[2] * x_hat.get('y').size()[3]
        nb_ex_in_batch = x_hat.get('y').size()[0]
        rate_bpp_y = rate_y.sum() / (nb_pixel * nb_ex_in_batch)
        rate_bpp_z = rate_z.sum() / (nb_pixel * nb_ex_in_batch)
        rate_bpp = rate_bpp_y + rate_bpp_z
        # ============ RATE ============ #

        final_loss = lambda_tradeoff * dist + rate_bpp

        return final_loss, dist, rate_bpp, rate_bpp_y, rate_bpp_z, mse,\
            ms_ssim, ms_ssim_db


class CustomL1Loss(Module):

    def __init__(self):
        super(CustomL1Loss, self).__init__()

    def forward(self, x_hat, code):
        """
        Compute the L1 norm between the decoder output <x_hat>
        and the reference frame <code>
        """
        x_hat_y, x_hat_u, x_hat_v = get_y_u_v(x_hat)
        code_y, code_u, code_v = get_y_u_v(code)

        l1_err = ((x_hat_y - code_y).abs()).sum()
        l1_err += ((x_hat_u - code_u).abs()).sum()
        l1_err += ((x_hat_v - code_v).abs()).sum()

        nb_values = x_hat_y.numel() + x_hat_u.numel() + x_hat_v.numel()
        l1_loss = l1_err / nb_values

        return l1_loss


class MSELoss(Module):

    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, x_hat, code):
        """
        Compute the MSE between the decoder output <x_hat>
        and the reference frame <code>
        """
        x_hat_y, x_hat_u, x_hat_v = get_y_u_v(x_hat)
        code_y, code_u, code_v = get_y_u_v(code)

        squarred_err = ((x_hat_y - code_y) ** 2).sum()
        squarred_err += ((x_hat_u - code_u) ** 2).sum()
        squarred_err += ((x_hat_v - code_v) ** 2).sum()

        nb_values = x_hat_y.numel() + x_hat_u.numel() + x_hat_v.numel()
        mse = squarred_err / nb_values

        return mse


class MSSSIMLoss(Module):

    def __init__(self):
        super(MSSSIMLoss, self).__init__()

        self.msssim_fn = MSSSIM(max_val=1.)

    def forward(self, x_hat, code):
        """
            Compute the MS-SSIM for each channel (Y, U, V) of
            x_hat (reconstructed) and code (gnd truth).

            MS-SSIM is weighted by the number of values in each channel.

        """

        x_hat_y, x_hat_u, x_hat_v = get_y_u_v(x_hat)
        code_y, code_u, code_v = get_y_u_v(code)

        nb_values = x_hat_y.numel() + x_hat_u.numel() + x_hat_v.numel()

        msssim = self.msssim_fn(x_hat_y, code_y) * x_hat_y.numel()
        # print('msssim with only y: ' + str(msssim))
        msssim += self.msssim_fn(x_hat_u, code_u) * x_hat_u.numel()
        # print('msssim with only y and u: ' + str(msssim))
        msssim += self.msssim_fn(x_hat_v, code_v) * x_hat_v.numel()
        # print('msssim with y, u and v: ' + str(msssim))

        msssim /= nb_values
        # We want to minimize something
        loss = 1 - msssim

        return loss, msssim
