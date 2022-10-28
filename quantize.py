import os
import copy
import torch
import logging

from torch import nn
import torch.nn.functional as F
from qatcnn import QatConv1d, QatConv2d, QatLinear

logger = logging.getLogger(__name__)

DEFAULT_MODULE_MAPPING = {
    nn.Linear: QatLinear,
    nn.Conv2d: QatConv2d,
}


# Modified from torch.quantization.quantize
def convert(module, mapping=DEFAULT_MODULE_MAPPING, inplace=True):
    r"""Converts the float module with observers (where we can get quantization
    parameters) to a quantized module.

    Args:
        module: calibrated module with observers
        mapping: a dictionary that maps from float module type to quantized
                 module type, can be overwrritten to allow swapping user defined
                 Modules
        qconfig: the default qconfig if module.qconfig does not exist
        inplace: carry out model transformations in-place, the original module
                 is mutated
    """
    if not inplace:
        module = copy.deepcopy(module)

    reassign = {}

    if type(module) in mapping:
        module = mapping[type(module)].from_float(module)

    for name, mod in module.named_children():
        # Use the attribute from current module to override
       # print("name, mod".format(name, mod))
        reassign[name] = convert(mod, mapping, inplace=True)

    for key, value in reassign.items():
        module._modules[key] = value

    return module


def freeze(qmodule, mapping=DEFAULT_MODULE_MAPPING, inplace=True):
    r"""Freeze the qat-module's parameters according to the attached qconfig.

    Args:
        qmodule: quantize aware training module
        inplace: carry out module transformations in-place, the original module
                 is mutated
    """

    if not inplace:
        qmodule = copy.deepcopy(qmodule)
    for module in qmodule.children():
        if type(module) in mapping:
            a = torch.fake_quantize_per_tensor_affine(module.weight.data.float(),
                                                      scale=2 ** (- 7), zero_point=0, quant_min=-128,
                                                      quant_max=127) * 128
            module.weight.data = a.char()
        else:
            module = freeze(module)
    return qmodule
