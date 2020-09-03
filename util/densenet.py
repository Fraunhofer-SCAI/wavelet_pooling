import math
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.wavelet_pool2d import StaticWaveletPool2d, AdaptiveWaveletPool2d
from util.learnable_wavelets import SoftOrthogonalWavelet, ProductFilter


def get_pool(pool_type):
    if pool_type == 'adaptive_wavelet':
        print('creating adaptive wavelet')
        degree = 1
        size = degree*2
        # wavelet = SoftOrthogonalWavelet(
        #             torch.rand(size, requires_grad=True)*2. - 1.,
        #             torch.rand(size, requires_grad=True)*2. - 1.,
        #             torch.rand(size, requires_grad=True)*2. - 1.,
        #             torch.rand(size, requires_grad=True)*2. - 1.)
        wavelet = ProductFilter(
                    torch.rand(size, requires_grad=True)*2. - 1.,
                    torch.rand(size, requires_grad=True)*2. - 1.,
                    torch.rand(size, requires_grad=True)*2. - 1.,
                    torch.rand(size, requires_grad=True)*2. - 1.)
        return AdaptiveWaveletPool2d(wavelet=wavelet)
    elif pool_type == 'wavelet':
        print('static wavelet')
        return StaticWaveletPool2d(wavelet=pywt.Wavelet('haar'))
    elif pool_type == 'max':
        print('max pool')
        return nn.MaxPool2d(2)
    elif pool_type == 'avg':
        print('static wavelet')
        return nn.AvgPool2d(2)
    else:
        raise NotImplementedError



class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0, pool_type='wavelet'):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
        self.pool = get_pool(pool_type)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return self.pool(out)

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class DenseNet3(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0,
                 pool_type='adaptive_wavelet'):
        super(DenseNet3, self).__init__()
        self.pool_type = pool_type
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n/2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate,
                                      pool_type=self.pool_type)
        in_planes = int(math.floor(in_planes*reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate,
                                      pool_type=self.pool_type)
        in_planes = int(math.floor(in_planes*reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)

    def get_wavelet_loss(self):
        if self.pool_type == 'adaptive_wavelet':
            return self.trans1.pool.wavelet.wavelet_loss() + \
                   self.trans2.pool.wavelet.wavelet_loss()
        else:
            return torch.tensor(0.)

    def get_pool(self):
        if self.pool_type == 'adaptive_wavelet':
            return [self.trans1.pool,
                    self.trans2.pool]
        else:
            return []

    def get_wavelets(self):
        if self.pool_type == 'adaptive_wavelet':
            return [self.trans1.pool.wavelet,
                    self.trans2.pool.wavelet]
        else:
            return []