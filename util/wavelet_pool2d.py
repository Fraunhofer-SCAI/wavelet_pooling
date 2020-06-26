import torch
import torch.nn as nn
from abc import ABC
from util.conv_transform import conv_fwt_2d, conv_ifwt_2d


class WaveletPool2d(nn.Module):
    """ Interface class for wavelet pooling objects."""

    def forward(self, img):
        fold_channels = torch.reshape(img, [img.shape[0]*img.shape[1],
                                            img.shape[2], img.shape[3]])
        fold_channels = fold_channels.unsqueeze(1)
        coeffs = conv_fwt_2d(fold_channels, wavelet=self.wavelet, scales=2)
        # rec = conv_ifwt_2d(coeffs, wavelet=self.wavelet)
        # rec = rec.reshape(img.shape)
        # err = torch.mean(torch.abs(img - rec))
        # print(err)
        pool = conv_ifwt_2d(coeffs[:-1], wavelet=self.wavelet)
        pool = pool.reshape([img.shape[0], img.shape[1],
                             pool.shape[-2], pool.shape[-1]])
        # remove wavelet padding.
        # pool = pool[..., :(img.shape[-2]//2), :(img.shape[-1]//2)]
        rescale = torch.mean(img)/torch.mean(pool)
        pool = rescale*pool
        return pool


class StaticWaveletPool2d(WaveletPool2d):
    def __init__(self, wavelet):
        super().__init__()
        self.wavelet = wavelet


class AdaptiveWaveletPool2d(WaveletPool2d):
    def __init__(self, wavelet):
        super().__init__()
        self.wavelet = wavelet
        assert self.wavelet.rec_lo.requires_grad is True, \
            'adaptive pooling requires grads.'
        assert self.wavelet.rec_hi.requires_grad is True
        assert self.wavelet.dec_lo.requires_grad is True
        assert self.wavelet.dec_hi.requires_grad is True

    def get_wavelet_loss(self):
        return self.wavelet.wavelet_loss()
