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


class FcAngleClassifier(ch.nn.Module):

    def __init__(self):
        super().__init__()
        self.avgpool = ch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = ch.nn.Linear(512, 512)
        self.fc2 = ch.nn.Linear(512, 2)
        self.extract_list = ['layer4']


    def forward(self, x: Tensor) -> (Tensor, Tensor):

        x = self.avgpool(x['layer4'])
        x = ch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x
