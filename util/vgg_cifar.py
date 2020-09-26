# Created by moritz (wolter@cs.uni-bonn.de) , 26.09.20
# based on https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from util.pool_select import get_pool
from util.wavelet_pool2d import AdaptiveWaveletPool2d, StaticWaveletPool2d

cfg = {
    'VGG11': [64, 'P', 128, 'P', 256, 256, 'P', 512, 512, 'P', 512, 512],
}


class VGG(nn.Module):
    def __init__(self, pool_type='adaptive_wavelet'):
        super().__init__()
        self.pool_type = pool_type
        # self.features = self._make_layers(cfg[vgg_name])
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool1 = get_pool(self.pool_type)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = get_pool(self.pool_type)
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = get_pool(self.pool_type)
        self.conv5 = torch.nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv6 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = get_pool(self.pool_type)
        self.conv7 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv8 = torch.nn.Conv2d(512, 512, kernel_size=2, padding=0)
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.pool3(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.pool4(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def get_wavelet_loss(self):
        if self.pool_type == 'adaptive_wavelet'\
            or self.pool_type == 'scaled_adaptive_wavelet':
            # loop trough the layers.
            wloss = self.pool1.get_wavelet_loss() \
                    + self.pool2.get_wavelet_loss() \
                    + self.pool3.get_wavelet_loss() \
                    + self.pool4.get_wavelet_loss()
            return wloss
        else:
            return torch.tensor(0.)

    def get_pool(self):
        if self.pool_type == 'adaptive_wavelet' \
           or self.pool_type == 'scaled_wavelet' \
           or self.pool_type == 'scaled_adaptive_wavelet':
            return [self.pool1, self.pool2, self.pool3, self.pool4]
        else:
            return []

    def get_wavelets(self):
        if self.pool_type == 'adaptive_wavelet' \
           or self.pool_type == 'scaled_adaptive_wavelet':
            return [self.pool1.wavelet, self.pool2.wavelet,
                    self.pool3.wavelet, self.pool4.wavelet]
        else:
            return []


def test():
    net = VGG()
    # print(net)
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())
    print(net.get_wavelet_loss())

if __name__ == '__main__':
    test()
