"""
FFCV transform that randomly rotates images
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
from torchvision.transforms.functional import rotate
from ffcv.pipeline.compiler import Compiler


class RandomRotate_Torch(Operation):
    """Transform using the given torch.nn.Module

    Parameters
    ----------
    module: torch.nn.Module
        The module for transformation
    """
    def __init__(self, individual_angle: bool = True):
        super().__init__()
        self.individual_angle = individual_angle


    def generate_code(self) -> Callable:
        if self.individual_angle:
            return self.generate_code_ind()
        return self.generate_code_block()


    def generate_code_block(self) -> Callable:
        def random_rotate_tensor(images, _):
            images = images.permute(0, 3, 1, 2)
            angle = int(np.random.randint(0, 360, size=1)[0])
            rotated = rotate(images, angle)

            # print("print")
            # from PIL import Image
            # Image.fromarray(np.array(rotated[1].cpu())).show()
            rotated = rotated.permute(0, 2, 3, 1)
            return rotated

        return random_rotate_tensor


    def generate_code_ind(self) -> Callable:
        parallel_range = Compiler.get_iterator()

        def random_rotate_tensor(images, _, indices):
            images = images.permute(0, 3, 1, 2)

            # _,_,h,w = images.shape
            # h, w = int(h/2), int(w/2)
            # images[:, :, h - 10:h, w - 10:w] = 0
            # images[:, :, h - 10:h, w : w + 10] = 125
            # images[:, :, h:h+10, w: w + 10] = 255
            # images[:, :, h:h + 10, w-10: w] = 125

            angle = np.random.randint(0, 360, size=images.shape[0])
            for i in parallel_range(len(indices)):
                images[i] = rotate(images[i], int(angle[i]))

            # print("print")
            # from PIL import Image
            # Image.fromarray(np.array(images[1].permute(1, 2, 0).cpu())).show()
            #Image.fromarray(rotate(images[1], int(angle[1])).permute(1,2,0).cpu()).show()
            images = images.permute(0, 2, 3, 1)
            return images

        random_rotate_tensor.is_parallel = True
        random_rotate_tensor.with_indices = True
        return random_rotate_tensor

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return previous_state, None
