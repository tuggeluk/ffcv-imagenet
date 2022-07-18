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
from torchvision.transforms.functional import rotate, InterpolationMode
from ffcv.pipeline.compiler import Compiler


class RandomRotate_Torch(Operation):
    """Transform using the given torch.nn.Module

    Parameters
    ----------
    module: torch.nn.Module
        The module for transformation
    """
    def __init__(self, block_rotate: bool = False, p_flip_upright = 0, double_rotate = False, pre_flip = False):
        super().__init__()
        self.block_rotate = block_rotate
        self.angle_config = -1
        self.p_flip_upright = p_flip_upright
        self.double_rotate = double_rotate
        self.pre_flip = pre_flip

    def set_angle_config(self, angle_config: int = -1, p_flip_upright = 0):
        self.angle_config = angle_config
        self.p_flip_upright = p_flip_upright


    def generate_code(self) -> Callable:
        if not self.block_rotate:
            return self.generate_code_ind()
        return self.generate_code_block()


    def generate_code_block(self) -> Callable:
        def random_rotate_tensor(images, _):
            images = images.permute(0, 3, 1, 2)
            if self.angle_config < 0:
                angle = int(np.random.randint(0, 360, size=1)[0])
            else:
                angle = self.angle_config

            images = rotate(images, angle)

            images = images.permute(0, 2, 3, 1)
            images = images.contiguous()
            return images

        return random_rotate_tensor


    def generate_code_ind(self) -> Callable:
        parallel_range = Compiler.get_iterator()

        def random_rotate_tensor(images, _, indices):

            if self.angle_config < 0:
                angle = np.random.randint(0, 360, size=images.shape[0])
                if self.p_flip_upright > 0:
                    upright = np.random.uniform(low=0, high=1, size=images.shape[0]) < self.p_flip_upright
                    angle[upright] = 0
            else:
                angle = np.ones(images.shape[0]) * self.angle_config

            # print("print")
            #
            # Image.fromarray(np.array(images[1].permute(1, 2, 0).cpu())).show()
            # Image.fromarray(np.array(rotate(images[1], 90).permute(1, 2, 0).cpu())).show()
            pre_rotate = np.zeros(images.shape[0])

            if self.pre_flip:

                flip_angs = np.random.choice([90,180,270], images.shape[0])

                for i in parallel_range(len(indices)):
                    flip_ang = flip_angs[i]
                    if flip_ang == 90:
                        # from PIL import Image
                        # Image.fromarray(np.array(images[1].transpose(0,1).flipud().cpu())).show()
                        # Image.fromarray(np.array(rotate(images[1].permute(2, 0, 1), 90).cpu().permute(1, 2, 0))).show()
                        images[i] = images[i].transpose(0,1).flipud()
                    if flip_ang == 180:
                        # Image.fromarray(np.array(images[1].flipud().fliplr().cpu())).show()
                        # Image.fromarray(np.array(rotate(images[1].permute(2,0,1), 180).cpu().permute(1, 2, 0))).show()
                        images[i] = images[i].flipud().fliplr()
                    if flip_ang == 270:
                        #Image.fromarray(np.array(images[1].transpose(0,1).fliplr().cpu())).show()
                        #Image.fromarray(np.array(rotate(images[1].permute(2, 0, 1), 270).cpu().permute(1, 2, 0))).show()
                        images[i] = images[i].transpose(0,1).fliplr()
                pre_rotate += flip_angs
            images = images.permute(0, 3, 1, 2)

            if self.double_rotate:
                pre_angle = np.random.randint(-360, 360, size=images.shape[0])
                for i in parallel_range(len(indices)):
                    images[i] = rotate(images[i], int(pre_angle[i]), interpolation=InterpolationMode.BILINEAR)
                pre_rotate += pre_angle

            #final rotation
            #compute offset
            fin_angle = (angle - pre_rotate)%360
            for i in parallel_range(len(indices)):
                images[i] = rotate(images[i], int(fin_angle[i]), interpolation=InterpolationMode.BILINEAR)



            images = images.permute(0, 2, 3, 1)
            out_tensor = ch.Tensor(angle).type(ch.int32)
            out_tensor = out_tensor.to(images.device)
            return images, out_tensor

        random_rotate_tensor.is_parallel = True
        random_rotate_tensor.with_indices = True
        return random_rotate_tensor

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return previous_state, None
