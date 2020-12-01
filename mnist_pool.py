import argparse
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard.writer import SummaryWriter
from util.wavelet_pool2d import StaticWaveletPool2d, AdaptiveWaveletPool2d
from util.learnable_wavelets import ProductFilter, SoftOrthogonalWavelet
from util.pool_select import get_pool


class Net(nn.Module):
    def __init__(self, pool_type):
        super(Net, self).__init__()
        self.pool_type = pool_type
        self.conv1 = nn.Conv2d(1, 20, 5, padding=0, stride=1)
        self.norm1 = nn.BatchNorm2d(20)
        self.pool1 = get_pool(self.pool_type, scales=3)
        self.conv2 = nn.Conv2d(20, 50, 5, padding=0, stride=1)
        self.norm2 = nn.BatchNorm2d(50)
        self.pool2 = get_pool(self.pool_type, scales=2)
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
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = self.norm1(x)
        x = self.conv2(x)
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)
        x = self.norm2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.lin(x)
        # x = self.norm4(x)
        output = F.log_softp(x, dim=1)
        return output

    def get_wavelet_loss(self):
        if self.pool_type == 'adaptive_wavelet'\
                or self.pool_type == 'scaled_adaptive_wavelet':
            return self.pool1.wavelet.wavelet_loss() + \
                   self.pool2.wavelet.wavelet_loss()
        else:
            return torch.tensor(0.)

    def get_pool(self):
        if self.pool_type == 'adaptive_wavelet' \
           or self.pool_type == 'scaled_wavelet' \
           or self.pool_type == 'scaled_adaptive_wavelet':
            return [self.pool1, self.pool2]
        else:
            return []

    def get_wavelets(self):
        if self.pool_type == 'adaptive_wavelet' \
           or self.pool_type == 'scaled_adaptive_wavelet':
            return [self.pool1.wavelet, self.pool2.wavelet]
        else:
            return []


class LeNet5(nn.Module):
    def __init__(self, pool_type):
        super(LeNet5, self).__init__()
        self.pool_type = pool_type
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6,
                            kernel_size=5, padding=2, stride=1)
        self.act1 = nn.ReLU()
        self.s2 = get_pool(pool_type, scales=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16,
                            kernel_size=5, padding=0, stride=1)
        self.act3 = nn.ReLU()
        self.s4 = get_pool(pool_type, scales=2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.c1(x)
        x = self.act1(x)
        x = self.s2(x)
        x = self.c3(x)
        x = self.act3(x)
        x = self.s4(x)
        x = x.reshape(-1, 16*5*5)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

    def get_wavelet_loss(self):
        if self.pool_type == 'adaptive_wavelet'\
                or self.pool_type == 'scaled_adaptive_wavelet':
            return self.s2.wavelet.wavelet_loss() + \
                   self.s4.wavelet.wavelet_loss()
        else:
            return torch.tensor(0.)

    def get_pool(self):
        if self.pool_type == 'adaptive_wavelet' \
           or self.pool_type == 'scaled_wavelet' \
           or self.pool_type == 'scaled_adaptive_wavelet':
            return [self.s2, self.s4]
        else:
            return []

    def get_wavelets(self):
        if self.pool_type == 'adaptive_wavelet' \
           or self.pool_type == 'scaled_adaptive_wavelet':
            return [self.s2.wavelet, self.s4.wavelet]
        else:
            return []


def train(args, writer, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        if model.pool_type == 'adaptive_wavelet'\
                or model.pool_type == 'scaled_adaptive_wavelet':
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

        # log to TensorBoard
        if args.tensorboard:
            #log_value('train_loss', losses.avg, epoch)
            writer.add_scalar('train_loss', loss, batch_idx * len(data) + epoch*len(train_loader.dataset))
            # log_value('train_acc', top1.avg, epoch)
            writer.add_scalar('wvl', wvl, batch_idx * len(data) + epoch*len(train_loader.dataset))

            if args.pooling_type == 'adaptive_wavelet' \
                or args.pooling_type == 'scaled_adaptive_wavelet':
                pool_layers = model.get_pool()
                for pool_no, pool in enumerate(pool_layers):
                    writer.add_scalar(
                        tag='train_wavelets_prod/ac_prod_filt_loss/pl_'
                            + str(pool_no),
                        scalar_value=pool.wavelet.pf_alias_cancellation_loss()[0],
                        global_step=batch_idx * len(data) + epoch*len(train_loader.dataset))
                    writer.add_scalar(
                        tag='train_wavelets_prod/ac_conv_loss/pl_'
                        + str(pool_no),
                        scalar_value=pool.wavelet.alias_cancellation_loss()[0],
                        global_step=batch_idx * len(data) + epoch*len(train_loader.dataset))
                    writer.add_scalar(
                        tag='train_wavelets_prod/pr_loss/pl_' + str(pool_no),
                        scalar_value=pool.wavelet.perfect_reconstruction_loss()[0],
                        global_step=batch_idx * len(data) + epoch*len(train_loader.dataset))
                    if type(pool.wavelet) is SoftOrthogonalWavelet:
                        writer.add_scalar(
                            tag='train_wavelets_orth/strang/pl_'
                                + str(pool_no),
                            scalar_value=pool.wavelet.rec_lo_orthogonality_loss(),
                            global_step=batch_idx * len(data) + epoch*len(train_loader.dataset))
                        writer.add_scalar(
                            tag='train_wavelets_orth/harbo/pl_' + str(pool_no),
                            scalar_value=pool.wavelet.filt_bank_orthogonality_loss(),
                            global_step=batch_idx * len(data) + epoch*len(train_loader.dataset))

            if args.pooling_type == 'adaptive_wavelet' \
                or args.pooling_type == 'scaled_wavelet' \
                or args.pooling_type == 'scaled_adaptive_wavelet':
                pool_layers = model.get_pool()
                for pool_no, pool in enumerate(pool_layers):
                    if pool.use_scale_weights is True:
                        for wno, weight in enumerate(pool.get_scales_weights()):
                            writer.add_scalar(
                                tag='train_wavelets_scales/weights/'
                                    + 'pl_' + str(pool_no) + '_no_' + str(wno),
                                scalar_value=weight,
                                global_step=batch_idx * len(data) + epoch*len(train_loader.dataset))
                        # print('stop')


def test(args, writer, epoch, model, device, test_loader):
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

    if args.tensorboard:
        # log_value('train_loss', losses.avg, epoch)
        writer.add_scalar('test_loss', test_loss, epoch)
        # log_value('train_acc', top1.avg, epoch)
        writer.add_scalar('test_correct', correct, epoch)
        writer.add_scalar('test_accuracy',
                          100. * correct / len(test_loader.dataset), epoch)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST learning')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=25, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--momentum', type=float, default=0.8, metavar='M',
                        help='LGD momentum')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--tensorboard', help='Log progress to TensorBoard',
                        action='store_true', default=False)
    parser.add_argument('--pooling_type', default='seperable_wavelet', type=str,
                        help='pooling type to use')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    print(args)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.tensorboard:
        # configure("runs/%s"%(args.name))
        writer = SummaryWriter(comment='_' + args.pooling_type
                                       + '_lr_' + str(args.lr)
                                       + '_g_' + str(args.gamma)
                                       + '_m_' + str(args.momentum))
    else:
        writer = None

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

    # model = Net(pool_type=args.pooling_type).to(device)
    model = LeNet5(pool_type=args.pooling_type).to(device)
    print('init wvl loss:', model.get_wavelet_loss())
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    # optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)

    if model.pool_type == 'adaptive_wavelet'\
        or model.pool_type == 'scaled_adaptive_wavelet':
        wavelets = model.get_wavelets()
        for wavelet in wavelets:
            print('init wvl loss', wavelet.wavelet_loss().item())
            for i in range(250):
                optimizer.zero_grad()
                wvl_loss = wavelet.wavelet_loss()
                wvl_loss.backward()
                # print(i, wvl_loss.item())
                optimizer.step()
            print('final wvl loss', wavelet.wavelet_loss().item())

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        if args.tensorboard:
            for param_group in optimizer.param_groups:
                writer.add_scalar('lr', param_group['lr'], epoch)

        train(args, writer, model, device, train_loader, optimizer,
              epoch=epoch)
        print('wvl loss:', model.get_wavelet_loss())
        test(args, writer, epoch, model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    print('done')


if __name__ == '__main__':
    main()
