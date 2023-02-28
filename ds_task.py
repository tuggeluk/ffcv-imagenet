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


def get_dataset(dir_path, name, image_set, transform):
    def sbd(*args, **kwargs):
        return torchvision.datasets.SBDataset(*args, mode="segmentation", **kwargs)

    paths = {
        "voc": (dir_path, torchvision.datasets.VOCSegmentation, 21),
        "voc_aug": (dir_path, sbd, 21),
        "coco": (dir_path, get_coco, 21),
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train):
    if train:
        return presets.SegmentationPresetTrain(base_size=520, crop_size=480)
    else:
        return presets.SegmentationPresetEval(base_size=520)


def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses["out"]

    return losses["out"] + 0.5 * losses["aux"]



def evaluate(model, data_loader, device, num_classes, preprocess, class_to_idx, model_amr):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    num_processed_samples = 0

    mean = torch.Tensor((0.485, 0.456, 0.406))
    std = torch.Tensor((0.229, 0.224, 0.225))

    conversion = T.ConvertImageDtype(torch.float)
    normalizer = T.Normalize(mean=mean, std=std)


    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, 100, header):

            de_normalized = (torch.permute(image, (0, 2, 3, 1)) * std + mean) * 256
            # from PIL import Image
            # Image.fromarray(de_normalized[0].type(torch.uint8).numpy()).show()
            # preprocess image





            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output["out"]

            confmat.update(target.flatten(), output.argmax(1).flatten())
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            num_processed_samples += image.shape[0]

        confmat.reduce_from_all_processes()

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    return confmat


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)


    print(args)

    device = torch.device(args.device)

    dataset_test, num_classes = get_dataset(args.data_path, args.dataset, "val", get_transform(False))


    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=0, collate_fn=utils.collate_fn
    )

    weights = FCN_ResNet50_Weights.DEFAULT
    model = fcn_resnet50(weights=weights)
    model.eval()
    model.to(device)

    model_amr = torch.load('ResNet50AMR300.chkp')
    model_amr.eval()


    preprocess = weights.transforms()
    class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}



    # We disable the cudnn benchmarking because it can noticeably affect the accuracy
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes, preprocess = preprocess,
                       class_to_idx = class_to_idx, model_amr=model_amr)
    print(confmat)




def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)

    parser.add_argument("--data-path", default="/home/tugg/Documents/rotation_module/coco_trainval2017", type=str, help="dataset path")
    parser.add_argument("--dataset", default="coco", type=str, help="dataset name")

    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")

    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")

    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)