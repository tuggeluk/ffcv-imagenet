from typing import Type, Union

import torch as ch
from torch import Tensor
import torch.nn as nn
from .base_angle_classifier import BaseAngleClassifier
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1
from torchvision.models.efficientnet import EfficientNet
from torchvision.models.vision_transformer import VisionTransformer, EncoderBlock
from functools import partial


class VitAngleClassifier(BaseAngleClassifier):

    def __init__(self, in_model, out_channels):
        super().__init__()


        if len(in_model.encoder.layers) == 24:
            self.base_in = 64
            self.extract_list = ["encoder_layer_0", "encoder_layer_7", "encoder_layer_15", "encoder_layer_23"]
            self.in_sizes = [1024, ]*4
            hidden_dim = 1024
            mlp_dim = 3072
            num_heads = 16

        elif len(in_model.encoder.layers) == 12:
            self.base_in = 64
            self.extract_list = ["encoder_layer_0", "encoder_layer_3", "encoder_layer_7", "encoder_layer_11"]
            self.in_sizes = [768, ]*4
            hidden_dim = 768
            mlp_dim = 3072
            num_heads = 16

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        dropout = 0.0
        attention_dropout = 0.0

        self.in_blocks = nn.ModuleList()
        for i in range(len(self.extract_list)):
            self.in_blocks.append(EncoderBlock(
                    num_heads,
                    hidden_dim,
                    mlp_dim,
                    dropout,
                    attention_dropout,
                    norm_layer,
                ))


        self.fuse_conv = nn.Conv1d(4, 1, kernel_size=1, stride=1, bias=False)
        self.fuse_conv2d = nn.Conv2d(4, 1, kernel_size=1, stride=1, bias=False)


        if isinstance(out_channels, list):
            self.fcs = nn.ModuleList()
            for out_chan in out_channels:
                self.fcs.append(ch.nn.Linear(hidden_dim, out_chan))
            self.multi_out = True
        else:
            self.fc = ch.nn.Linear(hidden_dim, out_channels)
            self.multi_out = False

    def forward(self, x: Tensor) -> (Tensor, Tensor):
        processed = list()

        for i in range(len(self.extract_list)):
            processed.append(x[self.extract_list[i]])

        processed = ch.stack(processed, 1)
        processed = self.fuse_conv2d(processed)
        processed = ch.squeeze(processed)

        for i in range(len(self.extract_list)):
            processed = self.in_blocks[i](processed)

        processed = processed[:, 0]

        if self.multi_out:
            x_out_list = list()
            for fc in self.fcs:
                x_out_list.append(fc(processed))
            x_out = tuple(x_out_list)
        else:
            x_out = self.fc(processed)

        return x_out


    # def forward(self, x: Tensor) -> (Tensor, Tensor):
    #     processed = list()
    #     for i in range(len(self.extract_list)):
    #         processed.append(self.in_blocks[i](x[self.extract_list[i]])[:, 0])
    #
    #     processed = ch.stack(processed, 1)
    #     processed = self.fuse_conv(processed)
    #     processed = ch.squeeze(processed)
    #
    #     if self.multi_out:
    #         x_out_list = list()
    #         for fc in self.fcs:
    #             x_out_list.append(fc(processed))
    #         x_out = tuple(x_out_list)
    #     else:
    #         x_out = self.fc(processed)
    #
    #     return x_out

    # def _create_downsample(self, in_size, out_size):
    #     if not in_size == out_size:
    #         return nn.Sequential(
    #             conv1x1(in_size, out_size, 1),
    #             self._norm_layer(out_size),
    #         )
    #     else:
    #         return nn.Identity()
    #

    # def _make_layer(
    #     self,
    #     block: Type[Union[BasicBlock, Bottleneck]],
    #     planes: int,
    #     blocks: int,
    #     stride: int = 1,
    #     dilate: bool = False,
    # ) -> nn.Sequential:
    #     norm_layer = self._norm_layer
    #     downsample = None
    #     previous_dilation = self.dilation
    #     if dilate:
    #         self.dilation *= stride
    #         stride = 1
    #     if stride != 1 or self.inplanes != planes * block.expansion:
    #         downsample = nn.Sequential(
    #             conv1x1(self.inplanes, planes * block.expansion, stride),
    #             norm_layer(planes * block.expansion),
    #         )
    #
    #     layers = []
    #     layers.append(
    #         block(
    #             self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
    #         )
    #     )
    #     self.inplanes = planes * block.expansion
    #     for _ in range(1, blocks):
    #         layers.append(
    #             block(
    #                 self.inplanes,
    #                 planes,
    #                 groups=self.groups,
    #                 base_width=self.base_width,
    #                 dilation=self.dilation,
    #                 norm_layer=norm_layer,
    #             )
    #         )
    #     return nn.Sequential(*layers)
