import argparse
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from util.wavelet_pool2d import StaticWaveletPool2d, AdaptiveWaveletPool2d
from util.learnable_wavelets import ProductFilter
# Test set: Average loss: 0.0295, Accuracy: 9905/10000 (99%)
# Wavelet Test set: Test set: Average loss: 0.0400, Accuracy: 9898/10000 (99%)
# maxPool: Test set: Average loss: 0.0216, Accuracy: 9944/10000 (99%)
# avgPool: 9865


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool_type = 'wavelet'

        def get_pool(pool_type):
            if pool_type == 'wavelet':
                print('wavelet pool')
                return StaticWaveletPool2d(wavelet=pywt.Wavelet('haar'))
            elif pool_type == 'adaptive_wavelet':
                # wavelet = ProductFilter(
                #     torch.tensor([0., 0., 0.7071067811865476,
                #                   0.7071067811865476, 0., 0.],
                #                  requires_grad=True),
                #     torch.tensor([0., 0., -0.7071067811865476,
                #                   0.7071067811865476, 0., 0.],
                #                  requires_grad=True),
                #     torch.tensor([0., 0., 0.7071067811865476,
                #                   0.7071067811865476, 0., 0.],
                #                  requires_grad=True),
                #     torch.tensor([0., 0., 0.7071067811865476,
                #                   -0.7071067811865476, 0., 0.],
                #                  requires_grad=True))
                wavelet = ProductFilter(
                    torch.tensor([0.7071067811865476,
                                  0.7071067811865476],
                                 requires_grad=True),
                    torch.tensor([-0.7071067811865476,
                                  0.7071067811865476],
                                 requires_grad=True),
                    torch.tensor([0.7071067811865476,
                                  0.7071067811865476],
                                 requires_grad=True),
                    torch.tensor([0.7071067811865476,
                                  -0.7071067811865476],
                                 requires_grad=True))
                return AdaptiveWaveletPool2d(wavelet=wavelet)
            elif pool_type == 'max':
                print('max pool')
                return nn.MaxPool2d(2)
            elif pool_type == 'avg':
                print('avg pool')
                return nn.AvgPool2d(2)
            else:
                raise NotImplementedError

        self.conv1 = nn.Conv2d(1, 20, 5, padding=0, stride=1)
        self.norm1 = nn.BatchNorm2d(20)
        self.pool1 = get_pool(self.pool_type)
        self.conv2 = nn.Conv2d(20, 50, 5, padding=0, stride=1)
        self.norm2 = nn.BatchNorm2d(50)
        self.pool2 = get_pool(self.pool_type)
        self.conv3 = nn.Conv2d(50, 500, 4, padding=0, stride=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(1)
        if self.pool_type == 'adaptive_wavelet':
            self.lin = nn.Linear(500, 10)
        else:
            self.lin = nn.Linear(500, 10)
        self.norm4 = nn.BatchNorm2d(10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.norm2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.lin(x)
        # x = self.norm4(x)
        output = F.log_softmax(x, dim=1)
        return output

    def get_wavelet_loss(self):
        if self.pool_type == 'adaptive_wavelet':
            return self.pool1.wavelet.wavelet_loss() + \
                   self.pool2.wavelet.wavelet_loss()
        else:
            return 0.


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        if model.pool_type == 'adaptive_wavelet':
            wvl = model.get_wavelet_loss()
            loss += wvl
        else:
            wvl = 0.

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:

            for param_group in optimizer.param_groups:
                lr = param_group['lr']

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \
                  \t lr: {:.6f} \t wvl: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), lr, wvl))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    print('init wvl loss:', model.get_wavelet_loss())
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        print('wvl loss:', model.get_wavelet_loss())
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    print('done')

if __name__ == '__main__':
    main()
