from typing import Type, Union

import torch as ch
from torch import Tensor
import torch.nn as nn
from .base_angle_classifier import BaseAngleClassifier
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1, ResNet



class VitAngleClassifierCNN(BaseAngleClassifier):

    def __init__(self, in_model, out_channels):
        super().__init__()


        if len(in_model.encoder.layers) == 24:
            self.extract_list = ["encoder_layer_0", "encoder_layer_7", "encoder_layer_15", "encoder_layer_23"]

        elif len(in_model.encoder.layers) == 12:
            self.extract_list = ["encoder_layer_0", "encoder_layer_3", "encoder_layer_7", "encoder_layer_11"]

        if isinstance(out_channels, list):
            print("only one output is supported")
            assert False

        self.fuse_conv = nn.Conv2d(4, 3, kernel_size=1, stride=1).cuda()
        self.angle_net = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=out_channels)


    def forward(self, x: Tensor) -> (Tensor, Tensor):
        processed = list()
        for i in range(len(self.extract_list)):
            processed.append(x[self.extract_list[i]])

        processed = ch.stack(processed, 1).type(ch.FloatTensor).cuda()
        processed = self.fuse_conv(processed)
        return self.angle_net(processed)



