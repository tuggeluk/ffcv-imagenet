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
                #dst[i] = ndimage.rotate(images[i], angle[i], reshape=False)
                rotation_mat = np.transpose(np.array([[np.cos(angle[i]), -np.sin(angle[i])],
                                                      [np.sin(angle[i]), np.cos(angle[i])]]))
                h, w, _ = images[i].shape

                pivot_point_x = np.floor(h/2)
                pivot_point_y = np.floor(w/2)

                new_img = np.zeros(images[i].shape)

                for height in range(h):  # h = number of row
                    for width in range(w):  # w = number of col
                        xy_mat = np.array([[width - pivot_point_x], [height - pivot_point_y]])

                        rotate_mat = np.dot(rotation_mat, xy_mat)

                        new_x = pivot_point_x + rotate_mat[0].astype(np.int32)
                        new_y = pivot_point_y + rotate_mat[1].astype(np.int32)

                        # if (0 <= new_x <= w - 1) and (0 <= new_y <= h - 1):
                        #     new_img[new_y, new_x] = images[i, height, width]
                dst[i] = new_img

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