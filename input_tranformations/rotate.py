"""
FFCV transform that randomly rotates images
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
from scipy import ndimage

class RandomRotate(Operation):
    def generate_code(self):
        parallel_range = Compiler.get_iterator()
        def rotate(images, dst):
            angle = np.random.randint(0, 306, size=images.shape[0])
            for i in parallel_range(images.shape[0]):
                    dst[i] = ndimage.rotate(images[i], angle[i], reshape=False)
            return dst

        rotate.is_parallel = True
        return rotate

    def declare_state_and_memory(self, previous_state):
        mem_allocation = AllocationQuery(previous_state.shape, previous_state.dtype)
        return (previous_state, mem_allocation)


# check if opencv is faster
# (h, w) = image.shape[:2]
# center = (w / 2, h / 2)
# angle = 30
# scale = 1
#
# M = cv2.getRotationMatrix2D(center, angle, scale)
# rotated = cv2.warpAffine(image, M, (w, h))