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
from torchvision.transforms.functional import rotate, InterpolationMode, resize
from ffcv.pipeline.compiler import Compiler
from PIL import Image, ImageDraw
from torchvision import transforms


class RandomRotate_Torch(Operation):
    """Transform using the given torch.nn.Module

    Parameters
    ----------
    module: torch.nn.Module
        The module for transformation
    """
    def __init__(self, block_rotate: bool = False, p_flip_upright = 0, double_rotate = False, pre_flip = False,
                 ret_orig_img = False, late_resize = -1, double_resize=0, load_noise = 0, interpolation = 1):
        super().__init__()
        self.block_rotate = block_rotate
        self.angle_config = -1
        self.p_flip_upright = p_flip_upright
        self.double_rotate = double_rotate
        self.pre_flip = pre_flip
        self.ret_orig_img = ret_orig_img
        self.late_resize = late_resize
        self.load_noise = load_noise
        self.interpolation = interpolation
        self.double_resize = double_resize

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

    def sw_rotate(self, image, angle):
        if self.interpolation == 0:
            image = rotate(image, int(angle), interpolation=InterpolationMode.NEAREST)
        elif self.interpolation == 1:
            image = rotate(image, int(angle), interpolation=InterpolationMode.BILINEAR)
        elif self.interpolation == 2:
            pil_image = Image.fromarray(image.cpu().numpy().transpose(1, 2, 0))
            pil_image = pil_image.rotate(int(angle), resample=Image.Resampling.BICUBIC)
            images = ch.tensor(np.array(pil_image).transpose(2,0,1))

        return image

    def generate_code_ind(self) -> Callable:
        parallel_range = Compiler.get_iterator()

        def random_rotate_tensor(images, _, indices):

            orig_imgs = None
            if self.ret_orig_img:
                orig_imgs = images.clone()




            if self.angle_config < 0:
                angle = np.random.randint(0, 360, size=images.shape[0])
                if self.p_flip_upright > 0:
                    upright = np.random.uniform(low=0, high=1, size=images.shape[0]) < self.p_flip_upright
                    angle[upright] = 0
            else:
                angle = np.ones(images.shape[0]) * self.angle_config

            if self.load_noise == 1:
                # store random noise in images
                images = (ch.rand(images.shape) * 255).type(ch.uint8).to(images.device)
            if self.load_noise == 2:
                # store random solid colors in images
                base_colors = (ch.rand((images.shape[-0], 1, 1, images.shape[-1])) * 255).type(ch.uint8)
                images[:] = base_colors[:]
            if self.load_noise == 3:
                # add random circles to blank page
                for i in parallel_range(len(indices)):
                    txt = Image.new("RGB", images.shape[1:3], (255, 255, 255))
                    d = ImageDraw.Draw(txt)
                    no_lines = np.random.randint(10)
                    for ii in range(no_lines):
                        d.line(tuple(np.random.randint(low=0, high=images.shape[1], size=4)),
                               fill=tuple(np.random.randint(255, size=3)),
                               width=np.random.randint(10))
                    images[i] = ch.tensor(np.array(txt))

            # print("print")
            # Image.fromarray(np.array(images[1].cpu())).show()
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
                if self.late_resize > 0 and self.double_resize == 1:
                    mid_size = np.random.randint(low=self.late_resize+5, high=images.shape[2]-5)
                    images = resize(images, interpolation=InterpolationMode.BICUBIC, size=mid_size,
                                    antialias=True)

                pre_angle = np.random.randint(-360, 360, size=images.shape[0])
                for i in parallel_range(len(indices)):
                    images[i] = self.sw_rotate(images[i], int(pre_angle[i]))
                pre_rotate += pre_angle

            #final rotation
            #compute offset
            fin_angle = (angle - pre_rotate)%360
            for i in parallel_range(len(indices)):
                images[i] = self.sw_rotate(images[i], int(fin_angle[i]))

            if self.late_resize > 0:
               images = resize(images, interpolation=InterpolationMode.BICUBIC, size=self.late_resize, antialias=True)

            images = images.permute(0, 2, 3, 1)
            out_tensor = ch.Tensor(angle).type(ch.int32)
            out_tensor = out_tensor.to(images.device)
            return (images, out_tensor, orig_imgs)

        random_rotate_tensor.is_parallel = True
        random_rotate_tensor.with_indices = True
        return random_rotate_tensor

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        if self.late_resize > 0:
            H, W, C = previous_state.shape
            return replace(previous_state, shape=(self.late_resize, self.late_resize, C)), None
        else:
            return previous_state, None
