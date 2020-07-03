import pywt
import numpy as np
import torch
from util.conv_transform import conv_fwt_2d, conv_ifwt_2d, get_pad
from util.conv_transform import flatten_2d_coeff_lst
from util.learnable_wavelets import SoftOrthogonalWavelet
from scipy import misc
import matplotlib.pyplot as plt
import tikzplotlib


iterations = 5000
face = misc.face()  # [128:(512+128), 256:(512+256)]
face = face / 255.
face = torch.tensor(face.astype(np.float32))
face = face.unsqueeze(0)
face = face.permute([3, 0, 1, 2])


def zero_by_scale(wavelet_coeffs, scale=5):
    '''
    Simply zero out entire scales.
    :param wavelet_coeffs: The list re
    :param zero_at:
    :return: A list where some coefficients are zeroed out.
    '''
    coefficients_low = []
    for no, c in enumerate(wavelet_coeffs):
        # print(scale, len(wavelet_coeffs) - no - 1, no)
        if scale <= len(wavelet_coeffs) - no - 1:
            coefficients_low.append(c)
        else:
            if type(c) is torch.Tensor:
                coefficients_low.append(0*c)
            elif type(c) is tuple:
                coefficients_low.append((0*c[0], 0*c[1], 0*c[2]))
            else:
                raise NotImplementedError
    return coefficients_low


wavelet = SoftOrthogonalWavelet(  # ProductFilter(
            torch.tensor([0, 0, 0.7071067811865476, 0.7071067811865476, 0, 0],
                         requires_grad=True),
            torch.tensor([0, 0, -0.7071067811865476, 0.7071067811865476, 0, 0],
                         requires_grad=True),
            torch.tensor([0, 0, 0.7071067811865476, 0.7071067811865476, 0, 0],
                         requires_grad=True),
            torch.tensor([0, 0, 0.7071067811865476, -0.7071067811865476, 0, 0],
                         requires_grad=True))


opt = torch.optim.Adagrad(wavelet.parameters(), lr=0.001)
coeffs = conv_fwt_2d(face, wavelet, scales=5)
z_coeffs = zero_by_scale(coeffs, scale=4)
init_rec = conv_ifwt_2d(z_coeffs, wavelet)
down_face = init_rec/torch.max(torch.abs(init_rec))
plt.imshow(down_face[:, 0, :, :].detach().permute([1, 2, 0]).numpy())
plt.show()

mse_list = []

for i in range(iterations):
    opt.zero_grad()
    coeffs = conv_fwt_2d(face, wavelet, scales=5)
    z_coeffs = zero_by_scale(coeffs, scale=4)
    rec = conv_ifwt_2d(z_coeffs, wavelet)
    diff = rec - face
    mse = torch.sum(diff*diff) + wavelet.wavelet_loss()
    mse.backward()
    opt.step()
    print(i,
          'mse', mse.detach().numpy(),
          'wvl', wavelet.wavelet_loss().detach().numpy())
    mse_list.append(mse.detach().numpy())

plt.plot(mse_list)
plt.show()

diff_s = diff*diff
diff_s = diff/torch.max(torch.abs(diff))
final = rec/torch.max(torch.abs(rec))
cat_imgs = torch.cat([final, down_face, diff_s], -1)
down_face = init_rec/torch.max(torch.abs(init_rec))
plt.imshow(cat_imgs[:, 0, :, :].detach().permute([1, 2, 0]).numpy())
plt.show()

print('optimization finished')

plt.imshow(final.detach().numpy())
plt.title('optimized-haar')
tikzplotlib.save('optimized_haar.tex', standalone=True)
plt.show()

plt.imshow(down_face.detach().numpy())
plt.title('haar')
tikzplotlib.save('optimized_haar.tex', standalone=True)
plt.show()

plt.imshow(diff_s.detach().numpy())
plt.title('normalized-difference')
tikzplotlib.save('normalized_difference.tex', standalone=True)
plt.show()

print('done')
