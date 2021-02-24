# Created by moritz (wolter@cs.uni-bonn.de), 14.04.20
import torch
import pywt


def get_filter_tensors(wavelet, flip, device):
    """Convert input wavelet to filter tensors.
    Args:
        wavelet: Wavelet object, assmuing ptwt-like
                 field names.
        flip ([bool]]): If true filters ar eflipped.
        device : PyTorch target device.
    Returns:
        Tuple containing the four filter tensors
        dec_lo, dec_hi, rec_lo, rec_hi
    """
    def create_tensor(filter):
        if flip:
            if isinstance(filter, torch.Tensor):
                return filter.flip(-1).unsqueeze(0).to(device)
            else:
                return torch.tensor(filter[::-1], device=device).unsqueeze(0)
        else:
            if isinstance(filter, torch.Tensor):
                return filter.unsqueeze(0).to(device)
            else:
                return torch.tensor(filter, device=device).unsqueeze(0)
    dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
    dec_lo = create_tensor(dec_lo)
    dec_hi = create_tensor(dec_hi)
    rec_lo = create_tensor(rec_lo)
    rec_hi = create_tensor(rec_hi)
    return dec_lo, dec_hi, rec_lo, rec_hi


def get_pad(data_len, filt_len):
    """ Compute the required padding.
    Args:
        data: The input tensor.
        wavelet: The wavelet filters used.
    Returns:
        The numbers to attach on the edges of the input.
    """
    # pad to ensure we see all filter positions and
    # for pywt compatability.
    # convolution output length:
    # see https://arxiv.org/pdf/1603.07285.pdf section 2.3:
    # floor([data_len - filt_len]/2) + 1
    # should equal pywt output length
    # floor((data_len + filt_len - 1)/2)
    # => floor([data_len + total_pad - filt_len]/2) + 1
    #    = floor((data_len + filt_len - 1)/2)
    # (data_len + total_pad - filt_len) + 2 = data_len + filt_len - 1
    # total_pad = 2*filt_len - 3

    # we pad half of the total requried padding on each side.

    # we pad half of the total requried padding on each side.
    padr = (2 * filt_len - 3) // 2
    padl = (2 * filt_len - 3) // 2

    # pad to even singal length.
    if data_len % 2 != 0:
        padl += 1

    return padr, padl


def fwt_pad(data, wavelet):
    """ Pad the input signal to make the fwt matrix work.
    Args:
        data: Input data [batch_size, 1, time]
        wavelet: The input wavelet following the pywt wavelet format.
    Returns:
        A pytorch tensor with the padded input data
    """
    padr, padl = get_pad(data.shape[-1], len(wavelet.dec_lo))

    # print('fwt pad', data.shape, pad)
    data_pad = torch.nn.functional.pad(data, [padl, padr],
                                       mode='reflect')
    return data_pad


def fwt_pad2d(data, wavelet):
    """Padding for the 2d FWT.
    Args:
        data (torch.Tensor): Input data with 4 domensions.
        wavelet (pywt.Wavelet or WaveletFilter): The wavelet used.
        mode (str, optional): [description]. Defaults to 'reflect'.
    Returns:
        The padded output tensor.
    """
    padb, padt = get_pad(data.shape[-2], len(wavelet.dec_lo))
    padr, padl = get_pad(data.shape[-1], len(wavelet.dec_lo))
    data_pad = torch.nn.functional.pad(data, [padt, padb, padl, padr],
                                       mode='reflect')
    return data_pad


def outer(a, b):
    """ Torch implementation of numpy's outer for vectors."""
    a_flat = torch.reshape(a, [-1])
    b_flat = torch.reshape(b, [-1])
    a_mul = torch.unsqueeze(a_flat, dim=-1)
    b_mul = torch.unsqueeze(b_flat, dim=0)
    return a_mul*b_mul


def flatten_2d_coeff_lst(coeff_lst_2d, flatten_tensors=True):
    flat_coeff_lst = []
    for coeff in coeff_lst_2d:
        if type(coeff) is tuple:
            for c in coeff:
                if flatten_tensors:
                    flat_coeff_lst.append(c.flatten())
                else:
                    flat_coeff_lst.append(c)
        else:
            if flatten_tensors:
                flat_coeff_lst.append(coeff.flatten())
            else:
                flat_coeff_lst.append(coeff)
    return flat_coeff_lst


def construct_2d_filt(lo, hi):
    """Construct two dimensional filters using outer
       products.
    Args:
        lo (torch.tensor): Low-pass input filter.
        hi (torch.tensor): High-pass input filter
    Returns:
        [torch.tensor]: Stacked 2d filters.
    """
    ll = outer(lo, lo)
    lh = outer(hi, lo)
    hl = outer(lo, hi)
    hh = outer(hi, hi)
    filt = torch.stack([ll, lh, hl, hh], 0)
    filt = filt.unsqueeze(1)
    return filt


def conv_fwt_2d(data, wavelet, scales: int = None) -> list:
    """ Non seperated two dimensional wavelet transform.

    Args:
        data (torch.tensor): [batch_size, 1, height, width]
        wavelet (WaveletFilter): The wavelet object to be used.
        scales (int, optional):  The scale level to be computed.
                                Defaults to None.

    Returns:
        [list]: List containing the wavelet coefficients.
    """
    # dec_lo, dec_hi, _, _ = wavelet.filter_bank
    # filt_len = len(dec_lo)
    # dec_lo = torch.tensor(dec_lo[::-1]).unsqueeze(0)
    # dec_hi = torch.tensor(dec_hi[::-1]).unsqueeze(0)
    dec_lo, dec_hi, _, _ = get_filter_tensors(wavelet, flip=True,
                                              device=data.device)
    # filt_len = dec_lo.shape[-1]
    dec_filt = construct_2d_filt(lo=dec_lo, hi=dec_hi)

    if scales is None:
        scales = pywt.dwtn_max_level([data.shape[-1], data.shape[-2]], wavelet)

    result_lst = []
    res_ll = data
    for s in range(scales):
        res_ll = fwt_pad2d(res_ll, wavelet)
        res = torch.nn.functional.conv2d(res_ll, dec_filt, stride=2)
        res_ll, res_lh, res_hl, res_hh = torch.split(res, 1, 1)
        result_lst.append((res_lh, res_hl, res_hh))
    result_lst.append(res_ll)
    return result_lst[::-1]


def conv_ifwt_2d(coeffs, wavelet):
    """Reconstruct a signal from wavelet coefficients.
    Args:
        coeffs (list): The wavelet coefficient list produced by wavedec2.
        wavelet (learnable_wavelets.WaveletFilter): The wavelet object
            used to compute the forward transform.
    Returns:
        torch.tensor: The reconstructed signal.
    """
    _, _, rec_lo, rec_hi = get_filter_tensors(
        wavelet, flip=False, device=flatten_2d_coeff_lst(coeffs)[0].device)
    filt_len = rec_lo.shape[-1]
    rec_filt = construct_2d_filt(rec_lo, rec_hi)

    res_ll = coeffs[0]
    for c_pos, res_lh_hl_hh in enumerate(coeffs[1:]):
        res_ll = torch.cat([res_ll, res_lh_hl_hh[0],
                            res_lh_hl_hh[1], res_lh_hl_hh[2]], 1)
        res_ll = torch.nn.functional.conv_transpose2d(res_ll, rec_filt,
                                                      stride=2)

        # remove the padding
        padl = (2*filt_len - 3)//2
        padr = (2*filt_len - 3)//2
        padt = (2*filt_len - 3)//2
        padb = (2*filt_len - 3)//2
        if c_pos < len(coeffs)-2:
            pred_len = res_ll.shape[-1] - (padl + padr)
            next_len = coeffs[c_pos+2][0].shape[-1]
            pred_len2 = res_ll.shape[-2] - (padt + padb)
            next_len2 = coeffs[c_pos+2][0].shape[-2]
            if next_len != pred_len:
                padl += 1
                pred_len = res_ll.shape[-1] - (padl + padr)
                assert next_len == pred_len, \
                    'padding error, please open an issue on github '
            if next_len2 != pred_len2:
                padt += 1
                pred_len2 = res_ll.shape[-2] - (padt + padb)
                assert next_len2 == pred_len2, \
                    'padding error, please open an issue on github '
        if padt > 0:
            res_ll = res_ll[..., padt:, :]
        if padb > 0:
            res_ll = res_ll[..., :-padb, :]
        if padl > 0:
            res_ll = res_ll[..., padl:]
        if padr > 0:
            res_ll = res_ll[..., :-padr]
    return res_ll


def conv_fwt(data, wavelet, scales: int = None) -> list:
    """Compute the analysis (forward) 1d fast wavelet transform."

    Args:
        data (torch.tensor): Input time series of shape [batch_size, 1, time]
        wavelet (learnable_wavelets.WaveletFilter): The wavelet object to be used.
        scales (int, optional): The scale level to be computed.
                                Defaults to None.

    Returns:
        [list]: A list containing the wavelet coefficients.
    """
    dec_lo, dec_hi, _, _ = get_filter_tensors(wavelet, flip=True,
                                              device=data.device)
    filt_len = dec_lo.shape[-1]
    # dec_lo = torch.tensor(dec_lo[::-1]).unsqueeze(0)
    # dec_hi = torch.tensor(dec_hi[::-1]).unsqueeze(0)
    filt = torch.stack([dec_lo, dec_hi], 0)

    if scales is None:
        scales = pywt.dwt_max_level(data.shape[-1], filt_len)

    result_lst = []
    res_lo = data
    for s in range(scales):
        res_lo = fwt_pad(res_lo, wavelet)
        res = torch.nn.functional.conv1d(res_lo, filt, stride=2)
        res_lo, res_hi = torch.split(res, 1, 1)
        result_lst.append(res_hi.squeeze(1))
    result_lst.append(res_lo.squeeze(1))
    return result_lst[::-1]


def conv_ifwt(coeffs: list, wavelet) -> torch.tensor:
    """Reconstruct a signal from wavelet coefficients.
    Args:
        coeffs (list): The wavelet coefficient list produced by wavedec.
        wavelet (learnable_wavelets.WaveletFilter): The wavelet object
            used to compute the forward transform.
    Returns:
        torch.tensor: The reconstructed signal.
    """
    _, _, rec_lo, rec_hi = get_filter_tensors(wavelet, flip=False,
                                              device=coeffs[-1].device)
    filt_len = rec_lo.shape[-1]

    filt = torch.stack([rec_lo, rec_hi], 0)

    res_lo = coeffs[0]
    for c_pos, res_hi in enumerate(coeffs[1:]):
        # print('shapes', res_lo.shape, res_hi.shape)
        res_lo = torch.stack([res_lo, res_hi], 1)
        res_lo = torch.nn.functional.conv_transpose1d(
            res_lo, filt, stride=2).squeeze(1)

        # remove the padding
        padl = (2*filt_len - 3)//2
        padr = (2*filt_len - 3)//2
        if c_pos < len(coeffs)-2:
            pred_len = res_lo.shape[-1] - (padl + padr)
            nex_len = coeffs[c_pos+2].shape[-1]
            if nex_len != pred_len:
                padl += 1
                pred_len = res_lo.shape[-1] - (padl + padr)
                assert nex_len == pred_len, \
                    'padding error, please open an issue on github '
            # ensure correct padding removal.
        # if padl > 0 and padr > 0:
        #     res_lo = res_lo[..., padl:-padr]
        if padl > 0:
            res_lo = res_lo[..., padl:]
        if padr > 0:
            res_lo = res_lo[..., :-padr]
    return res_lo
