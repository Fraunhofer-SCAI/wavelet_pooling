# Created by moritz wolter, 30.04.20
import torch
import pywt
from util.conv_transform import conv_fwt, conv_ifwt, conv_fwt_2d, conv_ifwt_2d


def get_layer_type(key_str, weight):
    """ Given a weight dict string find convolutional and fully connected weight tensors.
    :param key_str: The weight-dict key
    :return: conv, fc true for convolutional of fully connected layers.
    """
    wconv = False
    bconv = False
    wfc = False
    bfc = False
    key_parts = key_str.split('.')
    for kp in key_parts:
        if 'conv' in kp:
            for kp2 in key_parts:
                if 'weight' in kp2:
                    wconv = True
                elif 'bias' in kp2:
                    bconv = True
    for kp in key_parts:
        if 'shortcut' in kp:
            if type(weight) is tuple:
                wconv = True
            elif len(weight.shape) > 2:
                wconv = True
    for kp in key_parts:
        if ('fc' in kp) or ('linear' in kp):
            for kp2 in key_parts:
                if 'weight' in kp2:
                    wfc = True
                elif 'bias' in kp2:
                    bfc = True
    return wconv, wfc, bconv, bfc


def quantize_weight_dict(weight_dict, scale=0.01, zero_point=0):
    quant_dict = {}
    for key, weight in weight_dict.items():
        wconv, wfc, bconv, bfc = get_layer_type(key, weight)
        if type(weight) is torch.Tensor:
            if wconv or wfc or bconv or bfc:
                # print(key, weight.shape, wconv, wfc, bfc)
                quant_dict[key] = torch.quantize_per_tensor(weight, scale, zero_point, torch.qint8)
            else:
                quant_dict[key] = weight
        else:
            quant_dict[key] = weight
    return quant_dict


def dequantize_weight_dict(quantized_weight_dict):
    dequant_dict = {}
    for key, qweight in quantized_weight_dict.items():
        if type(qweight) is torch.Tensor:
            if qweight.is_quantized:
                dequant_dict[key] = qweight.dequantize()
            else:
                dequant_dict[key] = qweight
        else:
            dequant_dict[key] = qweight
    return dequant_dict


def wavelet_quantize_weight_dict(weight_dict, wavelet=pywt.Wavelet('db6'),
                                 fcwavelet=None, scale=0.01, zero_point=0, scales=None):
    if fcwavelet is None:
        fcwavelet = wavelet
    filter_len = len(wavelet.rec_hi)
    wquant_dict = {}
    for key, weight in weight_dict.items():
        shape = weight.shape
        wconv, wfc, bconv, bfc = get_layer_type(key, weight)
        if wconv:
            qcoeff = []
            wave1d = False
            if shape[0] < filter_len or shape[1] < filter_len:
                wave1d = True
            weightr = torch.reshape(weight, [shape[0], shape[1], -1])
            weight1rt = weightr.permute([-1, 0, 1])
            if wave1d:
                weight1rtr = torch.reshape(weight1rt, [shape[-1]*shape[-2], -1])
                coeff = conv_fwt(weight1rtr.unsqueeze(1), wavelet, scales=scales)
                for c in coeff:
                    qcoeff.append(torch.quantize_per_tensor(c, scale, zero_point, torch.qint8))
            else:
                coeff = conv_fwt_2d(weight1rt.unsqueeze(1), wavelet, scales=scales)
                for c in coeff:
                    if type(c) is tuple:
                        qcoeff.append((torch.quantize_per_tensor(c[0], scale, zero_point, torch.qint8),
                                       torch.quantize_per_tensor(c[1], scale, zero_point, torch.qint8),
                                       torch.quantize_per_tensor(c[2], scale, zero_point, torch.qint8)))
                    else:
                        qcoeff.append(torch.quantize_per_tensor(c, scale, zero_point, torch.qint8))
            wquant_dict[key] = (qcoeff, shape, wave1d)
        elif wfc:
            coeff = conv_fwt(weight.unsqueeze(1), fcwavelet, scales=scales)
            qcoeff = []
            for c in coeff:
                qcoeff.append(torch.quantize_per_tensor(c, scale, zero_point, torch.qint8))
            wquant_dict[key] = (qcoeff, shape, None)
        else:
            if wconv or wfc or bconv or bfc:
                wquant_dict[key] = torch.quantize_per_tensor(weight, scale, zero_point, torch.qint8)
            else:
                wquant_dict[key] = weight
    return wquant_dict


def dequantize_invwavelet_weight_dict(quantized_wavelet_dict, wavelet=pywt.Wavelet('db6'), fcwavelet=None):
    if fcwavelet is None:
        fcwavelet = wavelet
    rec_weight_dict = {}
    for key, quantized in quantized_wavelet_dict.items():
        # print(key)
        wconv, wfc, bconv, bfc = get_layer_type(key, quantized)
        if wconv:
            wave_lst, shape, wave1d = quantized
            dqcoeff_lst = []
            for qcoeff in wave_lst:
                if type(qcoeff) is tuple:
                    if type(qcoeff[0]) is torch.Size:
                        dqcoeff_lst.append((torch.zeros(qcoeff[0]), torch.zeros(qcoeff[1]), torch.zeros(qcoeff[2])))
                    elif qcoeff[0].is_quantized:
                        dqcoeff_lst.append((qcoeff[0].dequantize(), qcoeff[1].dequantize(), qcoeff[2].dequantize()))
                    else:
                        # do nothing.
                        dqcoeff_lst.append(qcoeff)
                else:
                    if type(qcoeff) is torch.Size:
                        dqcoeff_lst.append(torch.zeros(qcoeff))
                    elif qcoeff.is_quantized:
                        dqcoeff_lst.append(qcoeff.dequantize())
                    else:
                        # do nothing
                        dqcoeff_lst.append(qcoeff)
            if wave1d:
                synth_weightsrtr = conv_ifwt(dqcoeff_lst, wavelet)
                synth_weightsrt = torch.reshape(synth_weightsrtr, [shape[-1]*shape[-2], shape[0], shape[1]])
            else:
                synth_weightsrt = conv_ifwt_2d(dqcoeff_lst, wavelet).squeeze(1)
            synth_weightsr = synth_weightsrt.permute([1, 2, 0])
            rec_weight_dict[key] = torch.reshape(synth_weightsr, shape)
        elif wfc:
            dqcoeff_lst = []
            wave_lst, shape, wave1d = quantized
            for qcoeff in wave_lst:
                if type(qcoeff) is torch.Size:
                    dqcoeff_lst.append(torch.zeros(qcoeff))
                else:
                    if qcoeff.is_quantized:
                        dqcoeff_lst.append(qcoeff.dequantize())
                    else:
                        # do nothing.
                        dqcoeff_lst.append(qcoeff)
            synth_weights = conv_ifwt(dqcoeff_lst, fcwavelet)
            rec_weight_dict[key] = synth_weights
        else:
            if quantized.is_quantized:
                rec_weight_dict[key] = quantized.dequantize()
            else:
                rec_weight_dict[key] = quantized
    return rec_weight_dict


if __name__ == '__main__':
    mnist_weight_dict = torch.load('../mnist_cnn.pt', map_location=torch.device('cpu'))
    mnist_weight_dict_quant = torch.load('../mnist_quantized_cnn.pt', map_location=torch.device('cpu'))

    qdict = quantize_weight_dict(mnist_weight_dict)
    torch.save(qdict, '../qdict.pt')
    deq_dict = dequantize_weight_dict(qdict)

    wqdict = wavelet_quantize_weight_dict(mnist_weight_dict, pywt.Wavelet('db6'))
    torch.save(wqdict, '../wqdict.pt')
    # rswqdict = remove_scales(wqdict, cutoff_scale=1)
    # rswqdict = wqdict
    # torch.save(rswqdict, '../rwqdict.pt')
    # res_wqdict = dequantize_invwavelet_weight_dict(rswqdict, pywt.Wavelet('db6'))
    print('done')
