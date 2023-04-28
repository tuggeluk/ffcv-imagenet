# from torchvision.io.image import read_image
# from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
# from torchvision.transforms.functional import to_pil_image
#
# img = read_image("gallery/assets/dog1.jpg")
#
# # Step 1: Initialize model with the best available weights
# weights = FCN_ResNet50_Weights.DEFAULT
# model = fcn_resnet50(weights=weights)
# model.eval()
#
# # Step 2: Initialize the inference transforms
# preprocess = weights.transforms()
#
# # Step 3: Apply inference preprocessing transforms
# batch = preprocess(img).unsqueeze(0)
#
# # Step 4: Use the model and visualize the prediction
# prediction = model(batch)["out"]
# normalized_masks = prediction.softmax(dim=1)
# class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
# mask = normalized_masks[0, class_to_idx["dog"]]
# to_pil_image(mask).show()

import datetime
import os
import time
import warnings
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
import semseg_usecase.presets as presets
import torch
import torch.utils.data
import torchvision
import semseg_usecase.utils as utils
from semseg_usecase.coco_utils import get_coco
from torch import nn
from train_imagenet import BlurPoolConv2d
import torchvision.transforms as T

import numpy as np
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import functional as F, InterpolationMode
from torchvision.utils import draw_segmentation_masks
from PIL import Image


from torch.utils.data import Subset
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
from torchvision.datasets import CIFAR10, ImageFolder
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from typing import Union
from multiprocessing import set_start_method

from argparse import ArgumentParser
from fastargs import Section, Param
from fastargs.validation import And, OneOf
from fastargs.decorators import param, section
from fastargs import get_current_config


class RectImageFolder(ImageFolder):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )
        model_amr = torch.load('ResNet50AMR300.chkp')
        model_amr.eval()
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.amr_model = model_amr
        self.mean = torch.Tensor((0.485, 0.456, 0.406))
        self.std = torch.Tensor((0.229, 0.224, 0.225))

        self.conversion = T.ConvertImageDtype(torch.float)
        self.normalizer = T.Normalize(mean=self.mean, std=self.std)
        self.to_tens = T.ToTensor()


    def mask_corners(self, image):
        # assume square image
        x = np.arange(0, image.shape[0]) - np.floor(image.shape[0] / 2)
        y = np.arange(0, image.shape[0]) - np.floor(image.shape[0] / 2)
        xx, yy = np.meshgrid(x, y)
        mask = (np.sqrt((xx * xx) + (yy * yy)) - image.shape[0] / 2) > -3
        image[mask, :] = 0
        return image


    def de_normalize(self, image):
        return (torch.permute(image, (0, 2, 3, 1)) * self.std + self.mean) * 255


    def rectify_img(self, image):
        print("asdf")
        print(image.size)
        image = image.resize((256, 256), InterpolationMode.BILINEAR)

        im_masked = self.mask_corners(np.array(image))
        im_norm = self.normalizer(self.conversion(self.to_tens(im_masked)))

        output_cls, output_up, output_ang = self.amr_model(im_norm.unsqueeze(0).cuda())
        unrotate_ang = (360 - torch.argmax(output_ang, 1))%360

        if unrotate_ang[0] != 0:
            #print("test")
            im_corr = image.rotate(int(unrotate_ang[0]))
            #image.show()
            #im_corr.show()
        else:
            im_corr = image

        #de_norm = self.de_normalize(torch.unsqueeze(im_norm, 0))
        #Image.fromarray(de_norm[0].numpy().astype(np.uint8))

        return im_corr


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = self.rectify_img(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


def main(args):

    print(args)


    my_dataset = RectImageFolder(root=args.data_path)


    writer = DatasetWriter(args.data_out, {
        'image': RGBImageField(write_mode="smart",
                               max_resolution=256,
                               compress_probability=0.5,
                               jpeg_quality=90),
        'label': IntField(),
    }, num_workers=1)

    writer.from_indexed_dataset(my_dataset, chunksize=10)






def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)

    # parser.add_argument("--data_path", default="/cluster/data/amir/ImageNet/ImageNet1k/ImageNet/val", type=str, help="dataset path")
    # parser.add_argument("--data_out", default="/cluster/data/tugg/ImageNet_ffcv/val_rec.ffcv", type=str, help="path to save rectified datasets")
    parser.add_argument("--data_path", default="/media/tugg/sg_bak/Data_Archives/ImageNet_Debug/val", type=str,
                        help="dataset path")
    parser.add_argument("--data_out", default="/media/tugg/sg_bak/Data_Archives/trash_rec.ffcv", type=str,
                        help="path to save rectified datasets")

    return parser


if __name__ == "__main__":
    set_start_method('spawn', force=True)
    args = get_args_parser().parse_args()
    main(args)