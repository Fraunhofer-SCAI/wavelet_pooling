import pywt
import numpy as np
import torch
from util.conv_transform import conv_fwt_2d, conv_ifwt_2d, get_pad
from util.conv_transform import flatten_2d_coeff_lst
from util.learnable_wavelets import ProductFilter
from scipy import misc
import matplotlib.pyplot as plt
import tikzplotlib

scales=5
iterations = 1500
face = misc.face()  # [128:(512+128), 256:(512+256)]
face = face / 255.
face = torch.tensor(face.astype(np.float32))
face = face.unsqueeze(0)
face = face.permute([3, 0, 1, 2])


def set_scale(wavelet_coeffs, zero_scale=5, weights:list=None):
    '''
    Simply zero out entire scales.
    :param wavelet_coeffs: The list re
    :param zero_at:
    :return: A list where some coefficients are zeroed out.
    '''
    if weights is None:
        weights = torch.ones([len(wavelet_coeffs)])

    coefficients_low = []
    for no, c in enumerate(wavelet_coeffs):
        # print(scale, len(wavelet_coeffs) - no - 1, no)
        if zero_scale <= len(wavelet_coeffs) - no - 1:
            weight = weights[no]
        else:
            weight = 0.
        if type(c) is torch.Tensor:
            coefficients_low.append(weight*c)
        elif type(c) is tuple:
            coefficients_low.append((weight*c[0], weight*c[1], weight*c[2]))
        else:
            raise NotImplementedError

    return coefficients_low


wavelet = ProductFilter(
            torch.tensor([0, 0, 0.7071067811865476, 0.7071067811865476, 0, 0],
                         requires_grad=True),
            torch.tensor([0, 0, -0.7071067811865476, 0.7071067811865476, 0, 0],
                         requires_grad=True),
            torch.tensor([0, 0, 0.7071067811865476, 0.7071067811865476, 0, 0],
                         requires_grad=True),
            torch.tensor([0, 0, 0.7071067811865476, -0.7071067811865476, 0, 0],
                         requires_grad=True))

scale_weights = torch.ones(size=[scales], requires_grad=True) # + torch.rand([scales])

# opt = torch.optim.Adagrad(list(wavelet.parameters()) + [scale_weights], lr=0.001)
opt = torch.optim.Adam(list(wavelet.parameters()) + [scale_weights], lr=1e-03)
coeffs = conv_fwt_2d(face, wavelet, scales=scales)
z_coeffs = set_scale(coeffs, zero_scale=scales-1, weights=scale_weights)
init_rec = conv_ifwt_2d(z_coeffs, wavelet)
down_face = init_rec/torch.max(torch.abs(init_rec))
# plt.imshow(down_face[:, 0, :, :].detach().permute([1, 2, 0]).numpy())
# plt.show()

mse_list = []

for i in range(iterations):
    opt.zero_grad()
    coeffs = conv_fwt_2d(face, wavelet, scales=scales)
    z_coeffs = set_scale(coeffs, zero_scale=scales-1, weights=scale_weights)
    rec = conv_ifwt_2d(z_coeffs, wavelet)
    diff = rec - face
    mse = torch.sum(diff*diff) + wavelet.wavelet_loss()
    mse.backward()
    opt.step()
    print(i,
          'mse', mse.detach().numpy(),
          'wvl', wavelet.wavelet_loss().detach().numpy())
    mse_list.append(mse.detach().numpy())


plt.figure()
plt.plot(mse_list)
tikzplotlib.save('opt_mse.tex', standalone=True)
plt.show()

diff_s = (init_rec - rec)*(init_rec - rec)
diff_s = diff/torch.max(torch.abs(diff))
final = rec/torch.max(torch.abs(rec))
cat_imgs = torch.cat([final, down_face, diff_s], -1)
down_face = init_rec/torch.max(torch.abs(init_rec))
plt.imshow(cat_imgs[:, 0, :, :].detach().permute([1, 2, 0]).numpy())
plt.show()

print('optimization finished')
plt.figure()
plt.imshow(final[:, 0, :, :].permute([1, 2, 0]).detach().numpy())
plt.title('optimized-haar')
tikzplotlib.save('optimized_haar.tex', standalone=True)
plt.show()
plt.figure()
plt.imshow(down_face[:, 0, :, :].permute([1, 2, 0]).detach().numpy())
plt.title('haar')
tikzplotlib.save('haar.tex', standalone=True)
plt.show()
plt.figure()
plt.imshow(torch.mean(diff_s[:, 0, :, :], 0).detach().numpy(), cmap='Greys')
plt.title('normalized-difference')
tikzplotlib.save('normalized_difference.tex', standalone=True)
plt.show()

print('done')
