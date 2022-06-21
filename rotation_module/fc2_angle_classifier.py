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


class Fc2AngleClassifier(ch.nn.Module):

    def __init__(self, in_model):
        super().__init__()
        self.avgpool = ch.nn.AdaptiveAvgPool2d((1, 1))
        self.extract_list = ['layer4']
        self.in_size = self._get_recursive_last_size(in_model.get_submodule(self.extract_list[0]))
        self.fc1 = ch.nn.Linear(self.in_size, 512)
        self.fc2 = ch.nn.Linear(512, 2)

    def forward(self, x: Tensor) -> (Tensor, Tensor):

        x = self.avgpool(x['layer4'])
        x = ch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


    def _get_recursive_last_size(self, module):
        if len(module._modules) > 0:
            candidate = module._modules[list(module._modules.keys())[-1]]
            ret = self._get_recursive_last_size(candidate)
            if ret > 0:
                return ret
            else:
                for k,m in reversed(module._modules.items()):
                    if hasattr(m, "num_features"):
                        return m.num_features
        else:
            return -1