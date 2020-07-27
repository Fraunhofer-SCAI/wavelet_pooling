'''Train CIFAR10 with PyTorch.'''
# import pywt
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from util.learnable_wavelets import SoftOrthogonalWavelet
from util.helper_functions import progress_bar
# from util.wavelet_pool2d import StaticWaveletPool2d
from util.cifar_densenet import densenet_cifar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--pool', type=str, default='adaptive_wavelet',
                    help='Choose the pooling mode, adaptive_wavelet, wavelet, avg, max')
parser.add_argument('--log', action='store_true', default=True,
                    help='Turns on tensorboard logging.')


args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.log:
    writer = SummaryWriter(comment=args.pool)


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
print(args)
net = densenet_cifar(args.pool)
# print(net)
net = net.to(device)
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint_' + args.pool), 'No Checkpoint directory!'
    checkpoint = torch.load('./checkpoint_'
                            + args.pool + '/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    if args.pool == 'adaptive_wavelet':
        # pretrain wavelet.
        print('pretraining wavelets')
        wavelets = net.get_wavelets()
        for wavelet in wavelets:
            optimizer = optim.SGD(wavelet.parameters(), lr=0.01)
            print('init wvl loss', wavelet.wavelet_loss().item())
            for i in range(200):
                optimizer.zero_grad()
                wvl_loss = wavelet.wavelet_loss()
                wvl_loss.backward()
                # print(i, wvl_loss.item())
                optimizer.step()
            print('final wvl loss', wavelet.wavelet_loss().item())


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    wvl_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        closs = criterion(outputs, targets)
        loss = closs + net.get_wavelet_loss()
        loss.backward()
        optimizer.step()

        train_loss += closs.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if type(wvl_loss) == torch.tensor:
            wvl_loss += net.get_wavelet_loss().item()
        else:
            wvl_loss += net.get_wavelet_loss()

        progress_bar(batch_idx, len(trainloader),
                     'Train-Loss: %.3f | wvl %.5f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), wvl_loss/(batch_idx+1), 100.*correct/total,
                        correct, total))
        
        if args.log:
            n_iter = epoch*len(trainloader) + batch_idx
            # TODO add more to the tensorboard log.
            writer.add_scalar(tag='train/wvl_loss', scalar_value=net.get_wavelet_loss().item(), global_step=n_iter)
            writer.add_scalar(tag='train/cross_entropy_loss', scalar_value=closs.item(), global_step=n_iter)

            if args.pool == 'adaptive_wavelet':
                pool_layers = net.get_pool()
                for pool_no, pool in enumerate(pool_layers):
                    writer.add_scalar(tag='train_wavelets/ac_prod_filt_loss/w_' + str(pool_no),
                                      scalar_value=pool.wavelet.pf_alias_cancellation_loss()[0], global_step=n_iter)
                    writer.add_scalar(tag='train_wavelets/ac_conv_loss/w_' + str(pool_no),
                                      scalar_value=pool.wavelet.alias_cancellation_loss()[0],
                                      global_step=n_iter)
                    writer.add_scalar(tag='train_wavelets/pr_loss/w_' + str(pool_no),
                                      scalar_value=pool.wavelet.perfect_reconstruction_loss()[0],
                                      global_step=n_iter)

                    if type(pool.wavelet) is SoftOrthogonalWavelet:
                        writer.add_scalar(tag='train_wavelets/orth_strang/w_' + str(pool_no),
                                      scalar_value=pool.wavelet.rec_lo_orthogonality_loss(),
                                      global_step=n_iter)
                        writer.add_scalar(tag='train_wavelets/orth_harbo/w_' + str(pool_no),
                                      scalar_value=pool.wavelet.filt_bank_orthogonality_loss(),
                                      global_step=n_iter)

                    if pool.use_scale_weights is True:
                        writer.add_histogram('train_wavelets/scale_weights_'+ str(pool_no), values=pool.scales_weights,
                                             global_step=n_iter)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader),
                         'Test-Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1),
                            100.*correct/total, correct, total))


        if args.log:
            n_iter = epoch*len(trainloader) + batch_idx
            # TODO add more to the tensorboard log.
            writer.add_scalar(tag='test/acc', scalar_value=100.*correct/total, global_step=n_iter)
            writer.add_scalar(tag='test/loss', scalar_value=test_loss/batch_idx+1, global_step=n_iter)



    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint_' + args.pool):
            os.mkdir('checkpoint_' + args.pool)
        torch.save(state, './checkpoint_' + args.pool + '/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    # with torch.autograd.detect_anomaly():
    train(epoch)
    test(epoch)

print('done')