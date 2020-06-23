import torch
import torch.nn as nn
from util.conv_transform import conv_fwt_2d, conv_ifwt_2d


class StaticWaveletPool2d(nn.Module):
    def __init__(self, wavelet):
        super().__init__()
        self.wavelet = wavelet

    def forward(self, img):
        fold_channels = torch.reshape(img, [img.shape[0]*img.shape[1],
                                            img.shape[2], img.shape[3])
        fold_channels = img.unsqueeze(1)
        coeffs = conv_fwt_2d(fold_channels, self.wavelet, scales=2)
        rec = conv_ifwt_2d(coeffs)
        pool = conv_ifwt_2d(coeffs[-1])
        return img
