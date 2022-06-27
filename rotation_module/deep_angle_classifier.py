from typing import Type, Union

import torch as ch
from torch import Tensor
import torch.nn as nn
from .base_angle_classifier import BaseAngleClassifier
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1

class DeepAngleClassifier(BaseAngleClassifier):

    def __init__(self, in_model, out_channels, layers=(1, 1, 1, 1), flatten = 'basic'):
        super().__init__()

        # use ResNet Defaults
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        self.avgpool = ch.nn.AdaptiveAvgPool2d((1, 1))
        self.extract_list = ['maxpool', 'layer1', 'layer2', 'layer3', 'layer4']

        self.in_sizes = [self._get_recursive_last_size(in_model.get_submodule(x)) for x in self.extract_list if 'layer' in x]

        self.layer1 = self._make_layer(BasicBlock, self.in_sizes[0], layers[0])
        self.ds1 = self._create_downsample(self.in_sizes[0]*2, self.in_sizes[0])
        self.layer2 = self._make_layer(BasicBlock, self.in_sizes[1], layers[1], stride=2, dilate=False)
        self.ds2 = self._create_downsample(self.in_sizes[1]*2, self.in_sizes[1])
        self.layer3 = self._make_layer(BasicBlock, self.in_sizes[2], layers[2], stride=2, dilate=False)
        self.ds3 = self._create_downsample(self.in_sizes[2]*2, self.in_sizes[2])
        self.layer4 = self._make_layer(BasicBlock, self.in_sizes[3], layers[3], stride=2, dilate=False)
        self.ds4 = self._create_downsample(self.in_sizes[3]*2, self.in_sizes[3])

        if flatten == 'basic':
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            if isinstance(out_channels, list):
                self.fc1 = ch.nn.Linear(self.in_sizes[-1], out_channels[0])
                self.fc2 = ch.nn.Linear(self.in_sizes[-1], out_channels[1])
                self.multi_out = True
            else:
                self.fc = ch.nn.Linear(self.in_sizes[-1], out_channels)
                self.multi_out = False
        elif flatten == 'extended':
            self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
            if isinstance(out_channels, list):
                self.fc1 = ch.nn.Linear(self.in_sizes[-1]*25, out_channels[0])
                self.fc2 = ch.nn.Linear(self.in_sizes[-1]*25, out_channels[1])
                self.multi_out = True
            else:
                self.fc = ch.nn.Linear(self.in_sizes[-1]*25, out_channels)
                self.multi_out = False

        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x: Tensor) -> (Tensor, Tensor):

        x_out = self.layer1(x['maxpool'])
        x_out = self.ds1(ch.cat((x_out, x['layer1']), 1))
        x_out = self.layer2(x_out)
        x_out = self.ds2(ch.cat((x_out, x['layer2']), 1))
        x_out = self.layer3(x_out)
        x_out = self.ds3(ch.cat((x_out, x['layer3']), 1))
        x_out = self.layer4(x_out)
        x_out = self.ds4(ch.cat((x_out, x['layer4']), 1))

        x_out = self.avgpool(x_out)
        x_out = ch.flatten(x_out, 1)
        if self.multi_out:
            x_out = (self.fc1(x_out), self.fc2(x_out))
        else:
            x_out = self.fc(x_out)

        return x_out

    def _create_downsample(self, in_size, out_size):
        return nn.Sequential(
            conv1x1(in_size, out_size, 1),
            self._norm_layer(out_size),
        )


    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)
