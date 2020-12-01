# Created by moritz (wolter@cs.uni-bonn.de)
# I wrote this very quickly for the
# rebuttal its not pretty I don't recomment using it.
import torch
import numpy as np
import pywt

import os
print(os.path.abspath(os.getcwd()))

from util.conv_transform import get_pad, fwt_pad, fwt_pad2d
from util.conv_transform import conv_fwt, conv_fwt_2d
from util.conv_transform import get_filter_tensors
from util.conv_transform import construct_2d_filt
from util.conv_transform import flatten_2d_coeff_lst
from util.sparse_matmul_transform import matrix_fwt, matrix_ifwt


def sep_conv_fwt_2d(data, wavelet, scales: int = None) -> list:
    """ Non seperated two dimensional wavelet transform.

    Args:
        data (torch.tensor): [batch_size, height, width]
        wavelet (util.WaveletFilter or pywt.wavelet): The wavelet object.
        scales (int, optional): The number of decomposition scales.
                                 Defaults to None.

    Returns:
        [list]: List containing the wavelet coefficients.
    """
    ds = data.shape
    dec_lo, dec_hi, _, _ = get_filter_tensors(wavelet, flip=True,
                                              device=data.device)
    filt = torch.stack([dec_lo, dec_hi], 0)

    if scales is None:
        scales = pywt.dwtn_max_level([data.shape[-1], data.shape[-2]], wavelet)

    result_lst = []
    res_ll = data
    for s in range(scales):
        res_ll = fwt_pad2d(res_ll, wavelet)
        res_ll = res_ll.reshape([ds[0]*ds[1], res_ll.shape[2], res_ll.shape[3]])
        rll_s = res_ll.shape
        res_llr = res_ll.reshape([rll_s[0]*rll_s[1], rll_s[2]]).unsqueeze(1)
        res = torch.nn.functional.conv1d(res_llr, filt, stride=2)
        res_l, res_h = torch.split(res, 1, 1)
        res_l = res_l.reshape([rll_s[0], rll_s[1], rll_s[2]//2])
        res_h = res_h.reshape([rll_s[0], rll_s[1], rll_s[2]//2])
        res_lt = res_l.permute(0, 2, 1)
        res_ht = res_h.permute(0, 2, 1)
        res_ltr = res_lt.reshape([-1, res_lt.shape[-1]]).unsqueeze(1)
        res_htr = res_ht.reshape([-1, res_ht.shape[-1]]).unsqueeze(1)
        res_l2 = torch.nn.functional.conv1d(res_ltr, filt, stride=2)
        res_h2 = torch.nn.functional.conv1d(res_htr, filt, stride=2)
        res_ll, res_lh = torch.split(res_l2, 1, 1)
        res_hl, res_hh = torch.split(res_h2, 1, 1)
        res_llr = res_ll.reshape(rll_s[0], rll_s[2]//2, rll_s[1]//2)
        res_lhr = res_lh.reshape(rll_s[0], rll_s[2]//2, rll_s[1]//2)
        res_hlr = res_hl.reshape(rll_s[0], rll_s[2]//2, rll_s[1]//2)
        res_hhr = res_hh.reshape(rll_s[0], rll_s[2]//2, rll_s[1]//2)
        res_llrp = res_llr.permute([0, 2, 1])
        res_lhrp = res_lhr.permute([0, 2, 1])
        res_hlrp = res_hlr.permute([0, 2, 1])
        res_hhrp = res_hhr.permute([0, 2, 1])
        # res = torch.nn.functional.conv2d(res_ll, dec_filt, stride=2)
        # res_ll, res_lh, res_hl, res_hh = torch.split(res, 1, 1)
        result_lst.append(
            (res_lhrp.reshape(ds[0], ds[1], rll_s[1]//2, rll_s[2]//2),
             res_hlrp.reshape(ds[0], ds[1], rll_s[1]//2, rll_s[2]//2),
             res_hhrp.reshape(ds[0], ds[1], rll_s[1]//2, rll_s[2]//2)))
        res_ll = res_llrp.reshape(ds[0], ds[1], rll_s[1]//2, rll_s[2]//2)
    result_lst.append(res_llrp.reshape(ds[0], ds[1], rll_s[1]//2, rll_s[2]//2))
    return result_lst[::-1]


def inv_sep_conv_fwt_2d(coeffs, wavelet):
    _, _, rec_lo, rec_hi = get_filter_tensors(wavelet, flip=False,
                                              device=coeffs[0].device)
    filt = torch.stack([rec_lo, rec_hi], 0)

    res_ll = coeffs[0]
    for c_pos, res_lhhlhh in enumerate(coeffs[1:]):
        rll_s = res_ll.shape
        res_lh, res_hl, res_hh = res_lhhlhh
        res_ll = res_ll.reshape([rll_s[0]*rll_s[1], rll_s[2], rll_s[3]])
        res_lh = res_lh.reshape([rll_s[0]*rll_s[1], rll_s[2], rll_s[3]])
        res_hl = res_hl.reshape([rll_s[0]*rll_s[1], rll_s[2], rll_s[3]])
        res_hh = res_hh.reshape([rll_s[0]*rll_s[1], rll_s[2], rll_s[3]])

        res_ll = res_ll.permute([0, 2, 1])
        res_lh = res_lh.permute([0, 2, 1])
        res_hl = res_hl.permute([0, 2, 1])
        res_hh = res_hh.permute([0, 2, 1])
        res_ll = res_ll.reshape(-1, rll_s[2])
        res_lh = res_lh.reshape(-1, rll_s[2])
        res_hl = res_hl.reshape(-1, rll_s[2])
        res_hh = res_hh.reshape(-1, rll_s[2])
        res_l2 = torch.stack([res_ll, res_lh], 1)
        res_h2 = torch.stack([res_hl, res_hh], 1)
        res_l = torch.nn.functional.conv_transpose1d(res_l2, filt, stride=2)
        res_h = torch.nn.functional.conv_transpose1d(res_h2, filt, stride=2)
        res_lt = res_l.reshape([rll_s[0]*rll_s[1], rll_s[3], rll_s[2]*2])
        res_ht = res_h.reshape([rll_s[0]*rll_s[1], rll_s[3], rll_s[2]*2])
        res_l = res_lt.permute([0, 2, 1]).reshape(-1, rll_s[3])
        res_h = res_ht.permute([0, 2, 1]).reshape(-1, rll_s[3])
        res = torch.stack([res_l, res_h], dim=1)
        res_ll = torch.nn.functional.conv_transpose1d(res, filt, stride=2)
        res_ll = res_ll.reshape(rll_s[0], rll_s[1], rll_s[2]*2, rll_s[3]*2)
    return res_ll


if __name__ == '__main__':
    import warnings
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt
    print('gogogo')

    # ------------------------- 2d haar wavelet tests -----------------------
    import scipy.misc
    face = np.transpose(scipy.misc.face(), [2, 0, 1]).astype(np.float32)
    pt_face = torch.tensor(face).unsqueeze(1)
    wavelet = pywt.Wavelet('haar')

    # single level haar - 2d
    coeff2d_pywt = pywt.wavedec2(face, wavelet, level=2)
    coeff2d = conv_fwt_2d(pt_face, wavelet, scales=2)
    flat_lst = np.concatenate(flatten_2d_coeff_lst(coeff2d_pywt), -1)
    flat_lst2 = torch.cat(flatten_2d_coeff_lst(coeff2d), -1)
    err = np.mean(np.abs(flat_lst - flat_lst2.numpy()))
    print('haar 2d coeff err,', err, ['ok' if err < 1e-4 else 'failed!'])
    assert err < 1e-4

    test = sep_conv_fwt_2d(pt_face,
                           wavelet, scales=2)
    flat_lst3 = torch.cat(flatten_2d_coeff_lst(test), -1)
    err = np.mean(np.abs(flat_lst - flat_lst3.numpy()))
    print('sep haar 2d coeff err,', err, ['ok' if err < 1e-4 else 'failed!'])

    rec = inv_sep_conv_fwt_2d(test, wavelet)
    print('sep rec error:', torch.mean(torch.abs(pt_face - rec)).item())
    plt.imshow(
        rec[:, 0, :, :].permute(1, 2, 0).detach().numpy().astype(np.uint8))
    plt.show()
    print('done')
