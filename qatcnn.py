import torch
from torch import nn
import numpy as np
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import sys


def fake_quantize(data, total_bit, fraction_bit):
    """Fake quantize the data given on specific quantize_config"""
    if data is None:
        return None
    quant_range = 2 ** total_bit
    scale = 2 ** (- fraction_bit)
    quant_min = - quant_range // 2
    quant_max = quant_range // 2 - 1
    quantized_data = torch.fake_quantize_per_tensor_affine(
        data.float(),  # In case of amp auto_cast
        scale=scale, zero_point=0, quant_min=quant_min, quant_max=quant_max,
    )
    return quantized_data


class FakeQuantize(nn.Module):
    """Use float values to simulate quantized values

    See torch.nn.qat

    Args:
        - p(float): quantization noise ratio, see paper `Training with Quantization Noise for Extreme Model Compression`.
                    see fairseq for code reference.
    """

    def __init__(self, total_bit, fraction_bit):
        super().__init__()
        self.total_bit = total_bit
        self.fraction_bit = fraction_bit

    def extra_repr(self):
        return f'total_bit={self.total_bit}, fraction_bit={self.fraction_bit}'

    def forward(self, input):
        quantized_input = fake_quantize(input, self.total_bit, self.fraction_bit)
        return quantized_input


class QatConv1d(nn.Conv1d):

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.qat_input = FakeQuantize(8, 5)
        self.qat_weight = FakeQuantize(8, 7)
        # self.qat_weight = FakeQuantize(5,4)

    def forward(self, input):
        out = F.conv1d(self.qat_input(input), self.qat_weight(self.weight), self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
        return out

    @classmethod
    def from_float(cls, mod):
        qat_conv1d = cls(mod.in_channels, mod.out_channels, mod.kernel_size, mod.stride,
                         mod.padding, mod.dilation, mod.groups, bias=mod.bias is not None,
                         padding_mode=mod.padding_mode)

        qat_conv1d.weight = mod.weight
        qat_conv1d.bias = mod.bias
        return qat_conv1d


class QatConv2d(nn.Conv2d):

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        # self.qat_input = FakeQuantize(32, 28)  # (-8, 8)
        # self.qat_input = FakeQuantize(32, 29)  # (-4,4)
        # self.qat_input = FakeQuantize(32, 28) #(-8,8)
        self.qat_input = FakeQuantize(8, 4)
        # self.qat_weight = FakeQuantize(8, 7)
        self.qat_weight = FakeQuantize(8, 10)
        # self.qat_weight = FakeQuantize(32, 31)  # (-1,1)
        # self.qat_weight = FakeQuantize(32, 32)  # (-0.5, 0.5)
        # self.qat_weight = FakeQuantize(32, 33) #(-0.25,0.25)
        # self.qat_weight = FakeQuantize(32, 34)
        # self.qat_weight = FakeQuantize(6,5)

    def forward(self, input):
        out = F.conv2d(self.qat_input(input), self.qat_weight(self.weight), self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
        return out

    @classmethod
    def from_float(cls, mod):
        if mod.weight.requires_grad == False:
            return mod
        qat_conv2d = cls(mod.in_channels, mod.out_channels, mod.kernel_size, mod.stride, mod.padding,
                         mod.dilation, mod.groups, bias=mod.bias is not None, padding_mode=mod.padding_mode)

        qat_conv2d.weight = mod.weight
        qat_conv2d.bias = mod.bias
        return qat_conv2d


class QatLinear(nn.Linear):
    """Quantized Linear module

    This is a simplified version of torch's builtin qat.Linear module
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        # self.qat_input = FakeQuantize(32, 29)  # +-4    32，28（+-8）
        # self.qat_input = FakeQuantize(32, 28)
        self.qat_input = FakeQuantize(8, 4)
        # self.qat_weight = FakeQuantize(32, 31)  # +-1    32,32（+-0.5）
        # self.qat_weight = FakeQuantize(32, 32)
        # self.qat_weight = FakeQuantize(32, 33)
        # self.qat_weight = FakeQuantize(32, 34)
        # self.qat_weight = FakeQuantize(8, 7)
        self.qat_weight = FakeQuantize(8, 7)
        # self.qat_weight = FakeQuantize(5,4)

    def forward(self, input):
        output = F.linear(
            self.qat_input(input),
            self.qat_weight(self.weight),
            self.bias
        )
        return output

    @classmethod
    def from_float(cls, mod):
        if mod.weight.requires_grad == False:
            return mod
        qat_linear = cls(mod.in_features, mod.out_features, bias=mod.bias is not None)
        qat_linear.weight = mod.weight
        qat_linear.bias = mod.bias
        return qat_linear
