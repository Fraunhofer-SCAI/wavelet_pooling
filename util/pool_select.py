# Created by moritz (wolter@cs.uni-bonn.de) , 26.09.20
import pywt
import torch
import torch.nn as nn
from util.learnable_wavelets import ProductFilter
from util.wavelet_pool2d import AdaptiveWaveletPool2d, StaticWaveletPool2d


def get_pool(pool_type, scales=2):
    if pool_type == 'scaled_adaptive_wavelet':
        print('scaled adaptive wavelet')
        degree = 1
        size = degree*2
        wavelet = ProductFilter(
                    torch.rand(size, requires_grad=True)*2. - 1.,
                    torch.rand(size, requires_grad=True)*2. - 1.,
                    torch.rand(size, requires_grad=True)*2. - 1.,
                    torch.rand(size, requires_grad=True)*2. - 1.,)
        return AdaptiveWaveletPool2d(wavelet=wavelet,
                                     use_scale_weights=True,
                                     scales=scales)
    if pool_type == 'adaptive_wavelet':
        print('adaptive wavelet')
        degree = 1
        size = degree*2
        wavelet = ProductFilter(
                    torch.rand(size, requires_grad=True)*2. - 1.,
                    torch.rand(size, requires_grad=True)*2. - 1.,
                    torch.rand(size, requires_grad=True)*2. - 1.,
                    torch.rand(size, requires_grad=True)*2. - 1.)
        return AdaptiveWaveletPool2d(wavelet=wavelet,
                                     use_scale_weights=False,
                                     scales=scales)
    elif pool_type == 'wavelet':
        print('static wavelet')
        return StaticWaveletPool2d(wavelet=pywt.Wavelet('haar'),
                                   use_scale_weights=False,
                                   scales=scales)
    elif pool_type == 'seperable_wavelet':
        print('static seperable wavelet')
        return StaticWaveletPool2d(wavelet=pywt.Wavelet('haar'),
                                   use_scale_weights=False,
                                   scales=scales,
                                   seperable=True)
    elif pool_type == 'scaled_wavelet':
        print('scaled static wavelet')
        return StaticWaveletPool2d(wavelet=pywt.Wavelet('haar'),
                                   use_scale_weights=True,
                                   scales=scales)
    elif pool_type == 'max':
        print('max pool')
        return nn.MaxPool2d(2)
    elif pool_type == 'avg':
        print('avg pool')
        return nn.AvgPool2d(2)
    else:
        raise NotImplementedError
