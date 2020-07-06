import torch
import torch.nn as nn
from abc import ABC
from util.conv_transform import conv_fwt_2d, conv_ifwt_2d


class WaveletPool2d(nn.Module):
    """ Interface class for wavelet pooling objects."""

    def compute_padding(self):
        padr = 0
        padl = 0
        padt = 0
        padb = 0
        filt_len = len(self.wavelet.dec_lo)
        if filt_len > 2:
            padr += (2 * filt_len - 3) // 2
            padl += (2 * filt_len - 3) // 2
            padt += (2 * filt_len - 3) // 2
            padb += (2 * filt_len - 3) // 2
        return padr, padl, padt, padb

    def forward(self, img):
        fold_channels = torch.reshape(img, [img.shape[0]*img.shape[1],
                                            img.shape[2], img.shape[3]])
        fold_channels = fold_channels.unsqueeze(1)

        padr, padl, padt, padb = self.compute_padding()
        # TODO: add for higher degree wavelets.
        # fold_channels = torch.nn.functional_pad(fold_channels, [padt, padb, padl, padr])

        coeffs = conv_fwt_2d(fold_channels, wavelet=self.wavelet, scales=2)
        # rec = conv_ifwt_2d(coeffs, wavelet=self.wavelet)
        # rec = rec.reshape(img.shape)
        # err = torch.mean(torch.abs(img - rec))
        # print(err)
        pool = conv_ifwt_2d(coeffs[:-1], wavelet=self.wavelet)
        pool = pool.reshape([img.shape[0], img.shape[1],
                             pool.shape[-2], pool.shape[-1]])
        # remove wavelet padding.

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
