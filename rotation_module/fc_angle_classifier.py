import torch as ch
from torch import Tensor
from .base_angle_classifier import BaseAngleClassifier

class FcAngleClassifier(BaseAngleClassifier):

    def __init__(self, in_model, out_channels):
        super().__init__()
        self.avgpool = ch.nn.AdaptiveAvgPool2d((1, 1))
        self.extract_list = ['layer4']
        self.in_size = self._get_recursive_last_size(in_model.get_submodule(self.extract_list[0]))
        self.fc = ch.nn.Linear(self.in_size, out_channels)

    def forward(self, x: Tensor) -> (Tensor, Tensor):

        x = self.avgpool(x['layer4'])
        x = ch.flatten(x, 1)
        x = self.fc(x)

        return x
