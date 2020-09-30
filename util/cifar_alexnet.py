# created by moritz (wolter@cs.uni-bonn.de) 30.09.2020
# based on:
# https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
import torch
import torch.nn as nn
from util.pool_select import get_pool

class AlexNet(nn.Module):

    def __init__(self, pool_type='max', num_classes=10):
        super().__init__()
        self.pool_type = pool_type
        self.pool1 = get_pool(pool_type)
        self.pool2 = get_pool(pool_type)
        self.pool3 = get_pool(pool_type)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            self.pool1,
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            self.pool2,
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            self.pool3,
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_wavelet_loss(self):
        if self.pool_type == 'adaptive_wavelet'\
            or self.pool_type == 'scaled_adaptive_wavelet':
            return self.pool1.wavelet.wavelet_loss() + \
                   self.pool2.wavelet.wavelet_loss() + \
                   self.pool3.wavelet.wavelet_loss()    
        else:
            return torch.tensor(0.)

    def get_pool(self):
        if self.pool_type == 'adaptive_wavelet' \
           or self.pool_type == 'scaled_wavelet' \
           or self.pool_type == 'scaled_adaptive_wavelet':
            return [self.pool1, self.pool2, self.pool3]
        else:
            return []

    def get_wavelets(self):
        if self.pool_type == 'adaptive_wavelet' \
           or self.pool_type == 'scaled_adaptive_wavelet':
            return [self.pool1.wavelet, self.pool2.wavelet,
                    self.pool3.wavelet]
        else:
            return []
