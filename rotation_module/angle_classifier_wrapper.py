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
    def __init__(self, base_model, extract_list=['layer1', 'layer2', 'layer3', 'layer4']):
        super().__init__()
        self.base_model = base_model
        self.extract_list = extract_list
        self.avgpool = ch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = ch.nn.Linear(512, 512)
        self.fc2 = ch.nn.Linear(512, 2)



    def forward(self, x: Tensor) -> (Tensor, Tensor):

        extracted_tensors = dict()
        for name, mod in self.base_model._modules.items():
            if name == 'fc':
                x = ch.flatten(x, 1)
            x = mod(x)
            if name in self.extract_list:
                extracted_tensors[name] = x

        out = x
        out_wrapped = self.avgpool(extracted_tensors['layer4'])
        out_wrapped = ch.flatten(out_wrapped, 1)
        out_wrapped = self.fc1(out_wrapped)
        out_wrapped = self.fc2(out_wrapped)

        return out, out_wrapped