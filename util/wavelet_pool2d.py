import torch
import torch.nn as nn
# from abc import ABC
from util.conv_transform import conv_fwt_2d, conv_ifwt_2d
from util.sep_conv_transform import sep_conv_fwt_2d, inv_sep_conv_fwt_2d


class WaveletPool2d(nn.Module):
    """ Interface class for wavelet pooling objects."""

    def compute_padding(self):
        filt_len = len(self.wavelet.dec_lo)
        padr = (2 * filt_len - 3) // 2
        padl = (2 * filt_len - 3) // 2
        padt = (2 * filt_len - 3) // 2
        padb = (2 * filt_len - 3) // 2
        return padr, padl, padt, padb

    def forward(self, img):
        fold_channels = torch.reshape(img, [img.shape[0]*img.shape[1],
                                            img.shape[2], img.shape[3]])
        fold_channels = fold_channels.unsqueeze(1)

        padr, padl, padt, padb = self.compute_padding()

        # pad to even signal length
        if fold_channels.shape[-1] % 2 != 0:
            padl += 1
        if fold_channels.shape[-2] % 2 != 0:
            padt += 1

        fold_channels = torch.nn.functional.pad(fold_channels,
                                                [padt, padb, padl, padr])

        coeffs = self.conv_2d(fold_channels,
                              wavelet=self.wavelet,
                              scales=self.scales)

        pool_coeffs = []
        if self.use_scale_weights:
            weight_pos = 0
            for pos, coeff in enumerate(coeffs[:-1]):
                if type(coeff) is torch.Tensor:
                    pool_coeffs.append(
                        coeff*self.get_scales_weights()[weight_pos])
                    weight_pos += 1
                elif type(coeff) is tuple:
                    assert len(coeff) == 3, '2d fwt'
                    pool_coeffs.append(
                        (coeff[0]*self.get_scales_weights()[weight_pos],
                         coeff[1]*self.get_scales_weights()[weight_pos + 1],
                         coeff[2]*self.get_scales_weights()[weight_pos + 2]))
                    weight_pos += 3
        else:
            pool_coeffs = coeffs[:-1]

        pool = self.conv_2d_inverse(pool_coeffs, wavelet=self.wavelet)
        pool = pool.reshape([img.shape[0], img.shape[1],
                             pool.shape[-2], pool.shape[-1]])
        
        # remove wavelet padding.
        if pool.shape[-1] != img.shape[-1]//2:
            padl += 1
        if pool.shape[-2] != img.shape[-2]//2:
            padt += 1
        # print('pad', padr, padl, padt, padb)
        if padt > 0:
            pool = pool[..., padt:, :]
        if padb > 0:
            pool = pool[..., :-padb, :]
        if padl > 0:
            pool = pool[..., padl:]
        if padr > 0:
            pool = pool[..., :-padr]
        # pool = pool[..., :(img.shape[-2]//2), :(img.shape[-1]//2)]
        rescale = torch.mean(img)/torch.mean(pool)
        pool = rescale*pool
        return pool

    def get_scales_weights(self):
        if self.scales_weights is not None:
            return self.scales_weights*self.scales_weights


class StaticWaveletPool2d(WaveletPool2d):
    def __init__(self, wavelet, use_scale_weights=False, scales=3,
                 seperable=False):
        super().__init__()
        self.wavelet = wavelet
        self.use_scale_weights = use_scale_weights
        self.scales = scales
        if seperable:
            self.conv_2d = sep_conv_fwt_2d
            self.conv_2d_inverse = inv_sep_conv_fwt_2d
        else:
            self.conv_2d = conv_fwt_2d
            self.conv_2d_inverse = conv_ifwt_2d

        if self.use_scale_weights:
            weight_no = (self.scales - 1)*3 + 1
            self.scales_weights = torch.nn.Parameter(torch.ones([weight_no]))


class AdaptiveWaveletPool2d(WaveletPool2d):
    def __init__(self, wavelet, use_scale_weights=True, scales=3):
        super().__init__()
        self.wavelet = wavelet
        self.use_scale_weights = use_scale_weights
        self.scales = scales
        self.conv_2d = conv_fwt_2d
        self.conv_2d_inverse = conv_ifwt_2d
        assert self.wavelet.rec_lo.requires_grad is True, \
            'adaptive pooling requires grads.'
        assert self.wavelet.rec_hi.requires_grad is True
        assert self.wavelet.dec_lo.requires_grad is True
        assert self.wavelet.dec_hi.requires_grad is True

        if self.use_scale_weights:
            weight_no = (self.scales - 1)*3 + 1
            self.scales_weights = torch.nn.Parameter(torch.ones([weight_no]))

    def get_wavelet_loss(self):
        return self.wavelet.wavelet_loss()
