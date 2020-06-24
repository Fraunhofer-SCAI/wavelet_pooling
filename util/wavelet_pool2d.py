import torch
import torch.nn as nn
from util.conv_transform import conv_fwt_2d, conv_ifwt_2d, get_filter_tensors
from util.conv_transform import flatten_2d_coeff_lst, construct_2d_filt


class StaticWaveletPool2d(nn.Module):
    def __init__(self, wavelet):
        super().__init__()
        self.wavelet = wavelet

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
