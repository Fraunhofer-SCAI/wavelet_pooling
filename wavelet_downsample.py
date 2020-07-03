import pywt
import numpy as np
import torch
from util.conv_transform import conv_fwt_2d, conv_ifwt_2d, get_pad
from util.conv_transform import flatten_2d_coeff_lst
from scipy import misc
import matplotlib.pyplot as plt


face = misc.face() #[128:(512+128), 256:(512+256)]
face = face / 255.
face = torch.tensor(face.astype(np.float32))
face = face.unsqueeze(0)
face = face.permute([3, 0, 1, 2])
wavelet = pywt.Wavelet('db12')
coeff = conv_fwt_2d(face, wavelet=wavelet, scales=2)
print([c.shape for c in flatten_2d_coeff_lst(coeff)])
down_coeff = coeff[:-1]
print([c.shape for c in flatten_2d_coeff_lst(down_coeff)])
down_face = conv_ifwt_2d(down_coeff, wavelet=wavelet)
# TODO: Fix the padding problem!!!
filt_len = len(wavelet.dec_lo)

padr = 0
padl = 0
padt = 0
padb = 0
if filt_len > 2:
    padr += (2 * filt_len - 3) // 2
    padl += (2 * filt_len - 3) // 2
    padt += (2 * filt_len - 3) // 2
    padb += (2 * filt_len - 3) // 2
print('pad', padr, padl, padt, padb)
if padt > 0:
    down_face = down_face[..., padt:, :]
if padb > 0:
    down_face = down_face[..., :-padb, :]
if padl > 0:
    down_face = down_face[..., padl:]
if padr > 0:
    down_face = down_face[..., :-padr]
rescale = torch.mean(face)/torch.mean(down_face)
down_face = rescale*down_face
# down_face = down_face/torch.max(torch.abs(down_face))

print('face mean', np.mean(face.numpy()))
print('down face mean', np.mean(down_face.numpy()))

print('shape', down_face.shape)
plt.imshow(face.squeeze(1).permute([1, 2, 0]).numpy())
plt.show()
plt.imshow(down_face.permute([1, 2, 3, 0]).squeeze(0).numpy())
plt.show()

print('stop')
