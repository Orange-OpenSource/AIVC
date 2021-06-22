import torch
import torch.nn as nn
from torch.nn import Module, Parameter
from func_util.math_func import xavier_init, LOG_VAR_MIN, LOG_VAR_MAX
from func_util.console_display import print_log_msg
from torch.distributions import Normal, Laplace

class ParametricPdf(Module):

    def __init__(self, pdf_family):
        super(ParametricPdf, self).__init__()
        self.pdf_family = pdf_family

        print_log_msg(
            'INFO', 'ParametricPdf __init__', 'pdf_family', self.pdf_family
        )

    def forward(self, y_tilde, all_pdf_param, zero_mu=False):
        """
        If y_tilde shape is [B, M, H, W]
        mu and sigma are [B, M, H, W]

        all pdf_param is a list of dic gathering the parameters of each
        parametric model i.e. if we have a gaussian of 3 mixtures,
        we have len(all_pdf_param) = 3 and 3 dict of parameters.

        If zero_mu: set mu to an all-zero tensor (use this if we substract
        mu before quantization).
        """
        cur_device = y_tilde.device

        p_y_tilde = torch.zeros_like(y_tilde, device=cur_device)

        for pdf_param in all_pdf_param:
            # If True: mu has already been substracted
            if 'mu' in self.pdf_family.split('_') or zero_mu:
                mu = torch.zeros_like(y_tilde, device=cur_device)
            else:
                mu = pdf_param.get('mu')

            # To compute the scale_factor
            sigma = pdf_param.get('sigma')

            # p_y_tilde = torch.zeros(y_tilde.size(), device=cur_device)
            if 'normal' in self.pdf_family.split('_'):
                # For normal distribution, scale = sigma
                cur_pdf = Normal(mu, sigma)
            elif 'laplace' in self.pdf_family.split('_'):
                # For laplacian distribution, we need to compute the
                # scale parameters
                normalization = torch.sqrt(torch.tensor([2.0], device=cur_device))
                scale_factor = sigma / normalization
                cur_pdf = Laplace(mu, scale_factor)

            y_up = (y_tilde + 0.5)
            y_down = (y_tilde - 0.5)

            p_y_tilde += (cur_pdf.cdf(y_up) - cur_pdf.cdf(y_down))


        return p_y_tilde


class BallePdfEstim(nn.Module):
    """
    This module is a small neural network which has to learn the function
    p_x_tilde (x_tilde) which is the PDF of x_tilde = x * u.

    We want a forward method which returns p_x_tilde(x_tilde) so it is further
    optimized by minimizing -log p_x_tilde (which represents both rate and neg.
    log likelihood).

    However the architecture is made to represent a CDF, namely X (not x_tilde)
    CDF denoted c_x. Two reasons for this:
    - A cdf is easier to represent because of its inherent properties (cf.
    Ballé)

    - p_x_tilde (x_tilde) = c_x(x_tilde + 0.5) - c_x(x_tilde - 0.5)

    Therefore, there is two functions: cdf which computes c_x and forward which
    computes p_x_tilde.

    When used for inference, x_tilde = x_hat = Quantization(x). In this case,
    the forward pass return p_x_hat (the discrete quantization bins proba),
    needed for entropy coding.
    """

    def __init__(self, nb_channel, pdf_family, verbose=True):
        """
        Replicate the architecture and computation of "Variational Image
        Compression with a Scale Hyperprior", Ballé et al 2018 (Appendix 6)
        """
        super(BallePdfEstim, self).__init__()

        self.nb_channel = nb_channel
        self.pdf_family = pdf_family
        # Number of layers
        self.K = 4
        # Dimension of each hidden feature vector
        self.r = 3
        print_log_msg('INFO', '__init__ BallePdfEstim', 'K', self.K)
        print_log_msg('INFO', '__init__ BallePdfEstim', 'r', self.r)

        # Build Pdf Estimator Network
        self.matrix_h = nn.ParameterList()
        self.bias_b = nn.ParameterList()
        self.bias_a = nn.ParameterList()

        # We multiply by torch.sqrt(self.nb_channel) because xavier init
        # distribution have a variance of sqrt(2 / nb_weights) i.e:
        # sqrt(2 / (nb_channel * r_d * r_k)). However, we want to have
        # our matrix_h (3-d) as a 'list' of 2-d r_d * r_k matrix so
        # the normalisation factor is only sqrt(2 / (r_d * r_k))
        # Thus the multiplication by xav_correct
        xav_correc = torch.sqrt(torch.tensor([self.nb_channel]).float())

        for i in range(self.K):
            if i == 0:  # First Layer
                self.matrix_h.append(
                    nn.Parameter(
                        xavier_init((self.nb_channel, 1, self.r)) * xav_correc
                        )
                    )
                self.bias_a.append(
                    nn.Parameter(
                        xavier_init((self.nb_channel, self.r)) * xav_correc
                        )
                    )
                self.bias_b.append(
                    nn.Parameter(
                        xavier_init((self.nb_channel, self.r)) * xav_correc
                        )
                    )
            elif i == self.K - 1:  # Last layer
                self.matrix_h.append(
                    nn.Parameter(
                        xavier_init(
                            (self.nb_channel, self.r, 1)) * xav_correc
                        )
                    )
                self.bias_b.append(
                    nn.Parameter(
                        xavier_init((self.nb_channel, 1)) * xav_correc
                        )
                    )
            else:
                self.matrix_h.append(
                    nn.Parameter(
                        xavier_init((self.nb_channel, self.r, self.r)) * xav_correc
                        )
                    )
                self.bias_a.append(
                    nn.Parameter(
                        xavier_init((self.nb_channel, self.r)) * xav_correc
                        )
                    )
                self.bias_b.append(
                    nn.Parameter(
                        xavier_init((self.nb_channel, self.r)) * xav_correc
                        )
                    )

    def forward(self, x_tilde, pdf_param=None):
        """
        Compute p_x_tilde(x_tilde) = (p_x * U) (x_tilde)
                                   = cdf(x_tilde + 0.5) - cdf(x_tilde - 0.5)

        p_x_tilde (x_tilde) is a float tensor of shape:
            [B, C, H, W]
        (x_tilde has the same shape)
        Where:
            - B is the minibatch index
            - C is the features map index
        """
        cur_device = x_tilde.device
        # Scale factor replaces sigma if needed
        # TODO: Correct this, which no longer works with the new pdf_param
        if 'sigma' in self.pdf_family.split('_'):
            scale_factor = pdf_param[0].get('sigma') / torch.sqrt(torch.tensor([2.0], device=cur_device))
        else:
            scale_factor = torch.ones_like(x_tilde, device=cur_device)

        # Reshape x_tilde from [B, C, H, W] to
        # [B, C, H * W, 1]
        # Same for scale_factor
        B, C, H, W = x_tilde.size()
        x_tilde = x_tilde.view(B, C, H * W, 1)
        scale_factor = scale_factor.view(B, C, H * W, 1)
        p_x_tilde = self.cdf((x_tilde + 0.5) / scale_factor)\
            - self.cdf((x_tilde - 0.5) / scale_factor)

        # [B, C, H, W]
        return p_x_tilde.view(B, C, H, W)

    def cdf(self, x_tilde):
        # tmp_var is a placeholder for different calculation results throughout
        # the for loop.
        tmp_var = x_tilde
        for i in range(self.K):
            h_softplus = nn.functional.softplus(self.matrix_h[i])

            # h_softplus dimension is: [C, D, R]
            # tmp_var dimension is [B, C, E, X]
            # Where X is the nb of 'features' in the pdf estimator (i.e r)
            # When i == 0 ==> X = 1
            # Otherwise X = self.r

            # Perform Hx with a different H (2-d) matrix for each channel of x
            # tmp_var[i, :, :] = H[i, :, :] @ x [i, :, :]
            # Without the minibatch index

            # m: batch
            # c: channel
            # e: component in the c-th channel of the m-th minibatch
            # d and r: H goes from d to r
            tmp_var = torch.einsum('bced, cdr-> bcer', [tmp_var, h_softplus])
            # tmp_var dim is: [B, C, E, X]

            # bias_b dim is [C, X], so we repeat it * columns E times to obtain
            # bias_b [C, XE] where bias_b[:, X] == bias_b[:, X + kE]
            # We then reshape it to [C, E, X] (tmp_var.size()[1:])
            tmp_var += self.bias_b[i].repeat(1, tmp_var.size()[2]).view(tmp_var.size()[1:])

            # Non linearity is different for the last layer
            if i != self.K - 1:
                # Same thing than bias_b
                tmp_var = tmp_var + torch.mul(
                    torch.tanh(
                        self.bias_a[i].repeat(1, tmp_var.size()[2]).view(tmp_var.size()[1:])
                        ),
                    torch.tanh(tmp_var)
                    )
            else:
                p_x_tilde = torch.sigmoid(tmp_var)

        return p_x_tilde
