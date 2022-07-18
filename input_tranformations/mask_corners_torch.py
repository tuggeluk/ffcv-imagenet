"""
FFCV transform that masks the corners of an image, leaving a circle
"""

import torch as ch
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
from torchvision.transforms.functional import rotate, InterpolationMode
from ffcv.pipeline.compiler import Compiler


class MaskCorners_Torch(Operation):
    """Transform using the given torch.nn.Module

    Parameters
    ----------
    module: torch.nn.Module
        The module for transformation
    """
    def __init__(self):
        super().__init__()


    def generate_code(self) -> Callable:
        return self.generate_code()


    def generate_code(self) -> Callable:
        parallel_range = Compiler.get_iterator()

        def random_rotate_tensor(images, _, indices):
            # check for angles
            angles = None
            if isinstance(images, tuple):
                angles = images[1]
                images = images[0]
            #generate mask
            x = np.arange(0, images[0].shape[0], 1) - np.floor(images[0].shape[0] / 2)
            y = np.arange(0, images[0].shape[1], 1) - np.floor(images[0].shape[1] / 2)
            xx, yy = np.meshgrid(x, y)
            mask = (np.sqrt((xx * xx) + (yy * yy)) - images[0].shape[0] / 2) > 0

            # from PIL import Image
            # Image.fromarray(np.array(images[0].cpu())).show()
            images[:,mask,:] = 0


            if angles is not None:
                return (images, angles)
            return images

        random_rotate_tensor.is_parallel = True
        random_rotate_tensor.with_indices = True
        return random_rotate_tensor

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return previous_state, None
