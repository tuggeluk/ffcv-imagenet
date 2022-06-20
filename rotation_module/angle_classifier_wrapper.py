"""
FFCV transform that randomly rotates images
"""

import torch as ch
from torch import Tensor
import time
import numpy as np
import torchvision

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.state import State
from ffcv.pipeline.operation import Operation, AllocationQuery
from ffcv.transforms import ToTensor
from ffcv.writer import DatasetWriter
from dataclasses import replace
from numpy.random import permutation, rand
from typing import Callable, Optional, Tuple
from torchvision.transforms.functional import rotate
from ffcv.pipeline.compiler import Compiler


class AngleClassifierWrapper(ch.nn.Module):
    """Transform using the given torch.nn.Module

    Parameters
    ----------
    module: torch.nn.Module
        The module for transformation
    """
    def __init__(self, base_model, angle_class):
        super().__init__()
        self.base_model = base_model
        self.angle_class = angle_class

    def freeze_base(self) -> None:
        for param in self.base_model.parameters():
            param.requires_grad = False
        return None

    def unfreeze_base(self) -> None:
        for param in self.base_model.parameters():
            param.requires_grad = False
        return None

    def forward(self, x: Tensor) -> (Tensor, Tensor):

        extracted_tensors = dict()
        for name, mod in self.base_model._modules.items():
            if name == 'fc':
                x = ch.flatten(x, 1)
            x = mod(x)
            if name in self.angle_class.extract_list:
                extracted_tensors[name] = x

        out = x
        out_ang = self.angle_class(extracted_tensors)

        return out, out_ang