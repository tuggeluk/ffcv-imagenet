from typing import Type, Union

import torch as ch
from torch import Tensor
import torch.nn as nn
from .base_angle_classifier import BaseAngleClassifier
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1
from torchvision.models.efficientnet import EfficientNet

class DeepAngleClassifier(BaseAngleClassifier):

    def __init__(self, in_model, out_channels, layers=(1, 1, 1, 1), depths=(64, 128, 256, 512), flatten='basic'):
        super().__init__()

        # use ResNet Defaults
        self._norm_layer = nn.BatchNorm2d

        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.inplanes = 64

        if isinstance(in_model, EfficientNet):
            self.base_in = in_model.features._modules['0']._modules['0'].out_channels
            self.extract_list = ['0', '2', '4', '6', '8']
            self.in_sizes = [self._get_recursive_last_size(in_model.features._modules['2']),
                             self._get_recursive_last_size(in_model.features._modules['4']),
                             self._get_recursive_last_size(in_model.features._modules['6']),
                             self._get_recursive_last_size(in_model.features._modules['8'])]
            self.strides = [2, 4, 2, 1]

        else:
            self.base_in = 64
            self.extract_list = ['layer1', 'layer1', 'layer2', 'layer3', 'layer4']
            self.in_sizes = [self._get_recursive_last_size(in_model.get_submodule(x)) for x in self.extract_list if
                             'layer' in x]
            self.strides = [1, 2, 2, 2]

        self.avgpool = ch.nn.AdaptiveAvgPool2d((1, 1))

        self.dsmx_in = self._create_downsample(self.base_in, depths[0])

        self.layer1 = self._make_layer(BasicBlock, depths[0], layers[0], stride=self.strides[0])
        self.layer2 = self._make_layer(BasicBlock, depths[1], layers[1], stride=self.strides[1], dilate=False)
        self.layer3 = self._make_layer(BasicBlock, depths[2], layers[2], stride=self.strides[2], dilate=False)
        self.layer4 = self._make_layer(BasicBlock, depths[3], layers[3], stride=self.strides[3], dilate=False)

        # self.ds1_in = self._create_downsample(self.in_sizes[0], depths[0])
        # self.ds1_merge = self._create_downsample(depths[0]*2, depths[0])
        #
        # self.ds2_in = self._create_downsample(self.in_sizes[1], depths[1])
        # self.ds2_merge = self._create_downsample(depths[1]*2, depths[1])
        #
        # self.ds3_in = self._create_downsample(self.in_sizes[2], depths[2])
        # self.ds3_merge = self._create_downsample(depths[2]*2, depths[2])
        #
        # self.ds4_in = self._create_downsample(self.in_sizes[3], depths[3])
        # self.ds4_merge = self._create_downsample(depths[3]*2, depths[3])

        if flatten == 'basic':
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            if isinstance(out_channels, list):
                self.fcs = nn.ModuleList()
                for out_chan in out_channels:
                    self.fcs.append(ch.nn.Linear(depths[-1], out_chan))
                self.multi_out = True
            else:
                self.fc = ch.nn.Linear(depths[-1], out_channels)
                self.multi_out = False
        elif flatten == 'extended':
            self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
            if isinstance(out_channels, list):
                self.fcs =  nn.ModuleList()
                for out_chan in out_channels:
                    self.fcs.append(ch.nn.Linear(depths[-1], out_chan))
                self.multi_out = True
            else:
                self.fc = ch.nn.Linear(depths[-1]*25, out_channels)
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

        x_out = self.layer1(self.dsmx_in(x[self.extract_list[0]].type(ch.float)))
        #x_out = self.ds1_merge(ch.cat((x_out, self.ds1_in(x[self.extract_list[1]].type(ch.float))), 1))
        x_out = self.layer2(x_out)
        #x_out = self.ds2_merge(ch.cat((x_out, self.ds2_in(x[self.extract_list[2]].type(ch.float))), 1))
        x_out = self.layer3(x_out)
        #x_out = self.ds3_merge(ch.cat((x_out, self.ds3_in(x[self.extract_list[3]].type(ch.float))), 1))
        x_out = self.layer4(x_out)
        #x_out = self.ds4_merge(ch.cat((x_out, self.ds4_in(x[self.extract_list[4]].type(ch.float))), 1))

        x_out = self.avgpool(x_out)
        x_out = ch.flatten(x_out, 1)
        if self.multi_out:
            x_out_list = list()
            for fc in self.fcs:
                x_out_list.append(fc(x_out))
            x_out = tuple(x_out_list)
        else:
            x_out = self.fc(x_out)

        return x_out

    def _create_downsample(self, in_size, out_size):
        if not in_size == out_size:
            return nn.Sequential(
                conv1x1(in_size, out_size, 1),
                self._norm_layer(out_size),
            )
        else:
            return nn.Identity()


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
