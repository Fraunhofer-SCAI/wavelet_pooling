import pywt
import numpy as np
import torch
from util.conv_transform import conv_fwt_2d, conv_ifwt_2d
from scipy import misc
import matplotlib.pyplot as plt


face = misc.face()
face = face / 255.
face = torch.tensor(face.astype(np.float32))
face = face.unsqueeze(0)
face = face.permute([3, 0, 1, 2])

coeff = conv_fwt_2d(face, wavelet=pywt.Wavelet('haar'), scales=2)
down_coeff = coeff[:-1]
down_face = conv_ifwt_2d(down_coeff, wavelet=pywt.Wavelet('haar'))
rescale = torch.mean(face)/torch.mean(down_face)
down_face = rescale*down_face
# down_face = down_face/torch.max(torch.abs(down_face))

print('face mean', np.mean(face.numpy()))
print('down face mean', np.mean(down_face.numpy()))


plt.imshow(face.squeeze(1).permute([1, 2, 0]).numpy())
plt.show()
plt.imshow(down_face.permute([1, 2, 3, 0]).squeeze(0).numpy())
plt.show()

print('stop')
