"""
FFCV transform that randomly rotates images
"""
import time
import numpy as np
import torchvision
import torch.nn as nn

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.operation import Operation, AllocationQuery
from ffcv.transforms import ToTensor
from ffcv.writer import DatasetWriter
from dataclasses import replace
from scipy import ndimage

class RandomRotate_torch(nn.Module):

    def forward(self, x):
        return x


# check if opencv is faster
# (h, w) = image.shape[:2]
# center = (w / 2, h / 2)
# angle = 30
# scale = 1
#
# M = cv2.getRotationMatrix2D(center, angle, scale)
# rotated = cv2.warpAffine(image, M, (w, h))