"""
FFCV transform that masks corners, leaving a circle of unmasked pixels
"""
import time
import numpy as np
import torchvision

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.operation import Operation, AllocationQuery
from ffcv.transforms import ToTensor
from ffcv.writer import DatasetWriter
from dataclasses import replace


class MaskCorners(Operation):
    def generate_code(self):
        parallel_range = Compiler.get_iterator()
        def mask_corner(images, dst):
            for i in parallel_range(images.shape[0]):
                # x = np.arange(0, images[i].shape[0], 1) - np.floor(images[i].shape[0] / 2)
                # y = np.arange(0, images[i].shape[1], 1) - np.floor(images[i].shape[1] / 2)
                # xx = np.repeat(x,len(y)).reshape((len(y), len(x))).transpose()
                # yy = np.repeat(y,len(x)).reshape((len(y), len(x)))
                # mask = (np.sqrt((xx * xx) + (yy * yy)) - images[i].shape[0] / 2) > 0
                dst[i] = images[i]
                for y, dim1 in zip(np.arange(0, images[i].shape[0], 1) - np.floor(images[i].shape[0] / 2), range(images[i].shape[0])):
                    for x, dim2 in zip(np.arange(0, images[i].shape[1], 1) - np.floor(images[i].shape[1] / 2), range(images[i].shape[1])):
                        if (np.sqrt((x * x) + (y * y)) - images[i].shape[0] / 2) > 0:
                            dst[i, dim1, dim2] = 0
            return dst

        mask_corner.is_parallel = True
        return mask_corner

    def declare_state_and_memory(self, previous_state):
        mem_allocation = AllocationQuery(previous_state.shape, previous_state.dtype)
        return (previous_state, mem_allocation)