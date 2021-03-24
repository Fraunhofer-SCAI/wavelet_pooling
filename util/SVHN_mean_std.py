# Created by moritz (wolter@cs.uni-bonn.de) , 07.10.20

import numpy as np

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


transform = transforms.Compose([
    transforms.ToTensor(),
    ])

SVHN_train = datasets.SVHN('../../data', split='train', download=True,
                           transform=transform)
SVHN_test = datasets.SVHN('../../data', split='test', download=True,
                          transform=transform)
train_mean = np.mean(SVHN_train.data, axis=(0, 2, 3))
train_std = np.std(SVHN_train.data, axis=(0, 2, 3))
print('train_mean', train_mean)
print('train_std', train_std)


