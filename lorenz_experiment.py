import pywt
import numpy as np
import collections
import torch

import matplotlib.pyplot as plt
from util.mackey_glass import MackeyGenerator
from util.conv_transform import conv_fwt, conv_ifwt
from util.learnable_wavelets import ProductFilter, OrthogonalWavelet

CustomWavelet = collections.namedtuple('Wavelet', ['dec_lo', 'dec_hi',
                                                   'rec_lo', 'rec_hi', 'name'])
print(torch.cuda.is_available())


generator = MackeyGenerator(batch_size=64,
                            tmax=256.,
                            delta_t=0.1,
                            device='cpu')
mackey_data_1 = torch.squeeze(generator())

wavelet = pywt.Wavelet('haar')
# wavelet = pywt.Wavelet('db4')
# wavelet = pywt.Wavelet('bior2.4')

wavelet = CustomWavelet(
    dec_lo=[0, 0, 0.7071067811865476, 0.7071067811865476, 0, 0],
    dec_hi=[0, 0, -0.7071067811865476, 0.7071067811865476, 0, 0],
    rec_lo=[0, 0, 0.7071067811865476, 0.7071067811865476, 0, 0],
    rec_hi=[0, 0, 0.7071067811865476, -0.7071067811865476, 0, 0],
    name='custom')
wavelet = ProductFilter(torch.tensor(wavelet.dec_lo, requires_grad=True),
                        torch.tensor(wavelet.dec_hi, requires_grad=True),
                        torch.tensor(wavelet.rec_lo, requires_grad=True),
                        torch.tensor(wavelet.rec_hi, requires_grad=True))

# try out the multilevel version.
wave1d_8_freq = conv_fwt(mackey_data_1.unsqueeze(1).cpu(),
                         wavelet, scales=6)
print('alias cancellation loss:',
      wavelet.alias_cancellation_loss()[0].detach().numpy(), ',',
      'custom')
print('perfect reconstruction loss:',
      wavelet.perfect_reconstruction_loss()[0].detach().numpy())

# reconstruct the input
my_rec = conv_ifwt(wave1d_8_freq, wavelet)
print('my_rec error', np.sum(np.abs(my_rec[0, :].detach().numpy()
                                    - mackey_data_1[0, :].cpu().numpy())))

plt.plot(my_rec[0, :].detach().numpy())
plt.plot(mackey_data_1[0, :].cpu().numpy())
plt.plot(np.abs(my_rec[0, :].detach().numpy()
         - mackey_data_1[0, :].cpu().numpy()))
plt.show()


# wavelet compression.
# zero the low coefficients.
def zero_by_scale(wavelet_coeffs, zero_at=5):
    '''
    Simply zero out entire scales.
    :param wavelet_coeffs: The list re
    :param zero_at:
    :return: A list where some coefficients are zeroed out.
    '''
    coefficients_low = []
    for no, c in enumerate(wavelet_coeffs):
        if no > len(wave1d_8_freq) - zero_at:
            coefficients_low.append(c)
        else:
            coefficients_low.append(0*c)
    return coefficients_low


def zero_by_magnitude(wavelet_coeffs, cutoff_mag=5e-1):
    sparse_coefficients = []
    for no, c_vec in enumerate(wavelet_coeffs):
        mask = (torch.abs(c_vec) > cutoff_mag).type(torch.float32)
        c_vec_sparse = c_vec*mask
        sparse_coefficients.append(c_vec_sparse)
    return sparse_coefficients


zero_by_method = zero_by_magnitude

c_low = zero_by_method(wave1d_8_freq)

rec_low = conv_ifwt(c_low, wavelet=wavelet)
print('rec_low error', np.sum(np.abs(rec_low[0, :].detach().numpy()
                                     - mackey_data_1[0, :].cpu().numpy())))

plt.title('haar')
plt.plot(rec_low[0, :].detach().numpy())
plt.plot(mackey_data_1[0, :].cpu().numpy())
plt.plot(np.abs(rec_low[0, :].detach().numpy()
                - mackey_data_1[0, :].cpu().numpy()))
# savefig('haar')
# tikz.save('haar.tex', standalone=True)
plt.show()

plt.title('haar coefficients')
plt.semilogy(np.abs(torch.cat(wave1d_8_freq, -1)[0, :].detach().numpy()))
plt.semilogy(np.abs(torch.cat(c_low, -1)[0, :].detach().numpy()), '.')
# tikz.save('haar_coefficients.tex', standalone=True)
plt.show()

# optimize the basis:
steps = 2000
opt = torch.optim.Adagrad(wavelet.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

rec_loss_lst = []
for s in range(steps):
    opt.zero_grad()
    mackey_data = torch.squeeze(generator())
    wave1d_8_freq = conv_fwt(mackey_data.unsqueeze(1), wavelet,
                             scales=6)

    c_low = zero_by_method(wave1d_8_freq)

    rec_low = conv_ifwt(c_low, wavelet)
    msel = criterion(mackey_data, torch.squeeze(rec_low))
    loss = msel
    acl = wavelet.alias_cancellation_loss()[0]
    prl = wavelet.perfect_reconstruction_loss()[0]
    loss += (acl + prl)  # * s/steps

    # compute gradients
    loss.backward()
    # apply gradients
    opt.step()
    rec_loss_lst.append(msel.detach().cpu().numpy())
    if s % 250 == 0:
        print(s, loss.detach().cpu().numpy(),
              'mse', msel.detach().cpu().numpy(),
              'acl', acl.detach().cpu().numpy(),
              'prl', prl.detach().cpu().numpy())


wave1d_8_freq = conv_fwt(mackey_data_1.unsqueeze(1), wavelet, scales=6)

c_low = zero_by_method(wave1d_8_freq)

rec_low = conv_ifwt(c_low, wavelet)
print('rec_low error', np.sum(np.abs(rec_low[0, :].detach().cpu().numpy()
                                     - mackey_data_1[0, :].cpu().numpy())))

plt.title('Optimized Haar')
plt.plot(rec_low[0, :].detach().cpu().numpy())
plt.plot(mackey_data_1[0, :].cpu().numpy())
plt.plot(np.abs(rec_low[0, :].detach().cpu().numpy() - mackey_data_1[0, :].cpu().numpy()))
# plt.savefig('optimized_haar')
# tikz.save('optimized_haar.tex', standalone=True)
plt.show()

plt.title('Optimized haar coefficients')
plt.semilogy(np.abs(torch.cat(wave1d_8_freq, -1)[0, :].detach().cpu().numpy()))
plt.semilogy(np.abs(torch.cat(c_low, -1)[0, :].detach().cpu().numpy()), '.')
# tikz.save('optimized_haar_coefficients.tex', standalone=True)
plt.show()

plt.semilogy(rec_loss_lst[10:])
plt.show()
