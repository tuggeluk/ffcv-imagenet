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


def save_visualizations(input, output, groundtruth, name_prefix, path="semseg_usecase/images_out"):

    def generate_all_classes_mask(output):
        normalized_masks = torch.nn.functional.softmax(output, dim=1)
        num_classes = normalized_masks.shape[1]
        masks = normalized_masks[0]
        class_dim = 0
        all_classes_masks = masks.argmax(class_dim) == torch.arange(num_classes)[:, None, None]
        return all_classes_masks


    Image.fromarray(input[0].numpy().astype(np.uint8)).save(os.path.join(path, name_prefix+'inp.png'))

    input = torch.permute(input[0], (2, 0, 1)).type(torch.uint8)

    all_classes_masks_pred = generate_all_classes_mask(output)
    with_all_masks_pred = draw_segmentation_masks(input, masks=all_classes_masks_pred, alpha=.6)
    F.to_pil_image(with_all_masks_pred).save(os.path.join(path, name_prefix+'pred.png'))

    # groundtruth_back = groundtruth.clone()
    # groundtruth = groundtruth_back.clone()

    groundtruth[groundtruth == 255] = 0
    groundtruth = torch.nn.functional.one_hot(groundtruth.squeeze().long(), num_classes=21).bool().permute(2, 0, 1)
    with_all_masks_gt = draw_segmentation_masks(input, masks=groundtruth, alpha=.6)
    F.to_pil_image(with_all_masks_gt).save(os.path.join(path, name_prefix+'gt.png'))



def evaluate(model, data_loader, device, num_classes, preprocess, class_to_idx, model_amr):
    model.eval()
    confmat_upright = utils.ConfusionMatrix(num_classes)
    confmat_random_rot = utils.ConfusionMatrix(num_classes)
    confmat_amr = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    num_processed_samples = 0

    mean = torch.Tensor((0.485, 0.456, 0.406))
    std = torch.Tensor((0.229, 0.224, 0.225))

    conversion = T.ConvertImageDtype(torch.float)
    normalizer = T.Normalize(mean=mean, std=std)

    def de_normalize(image):
        return (torch.permute(image, (0, 2, 3, 1)) * std + mean) * 256

    def random_rotate(image, angle=-1, interpolation=InterpolationMode.BILINEAR):
        image = torch.permute(image, (0,3,1,2))
        if angle == -1:
            angle = np.random.randint(0, 359)

        image = F.rotate(image, int(angle), interpolation=interpolation)
        image = torch.permute(image, (0,2,3,1))
        return image

    # hardcode mask size for speedup
    x = np.arange(0, 520, 1) - np.floor(520 / 2)
    y = np.arange(0, 520, 1) - np.floor(520 / 2)
    xx, yy = np.meshgrid(x, y)
    mask = (np.sqrt((xx * xx) + (yy * yy)) - 520 / 2) > -3
    def mask_corners(image):
        image[:, mask ,:] = 0
        return image

    def normalize(image):
        return normalizer(conversion(torch.permute(image,(0,3,1,2))/256))

    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, 100, header):

            image_upright = image.clone()
            image_upright = de_normalize(image_upright)
            image_upright = mask_corners(image_upright)
            image_upright = normalize(image_upright)

            target_upright = mask_corners(torch.permute(target.clone(), (0,2,3,1)))

            image_upright, target_upright = image_upright.to(device), target_upright.to(device)
            output_upright = model(image_upright)
            output_upright = output_upright["out"]

            confmat_upright.update(target_upright.flatten(), output_upright.argmax(1).flatten())
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            save_visualizations(de_normalize(image_upright.detach().cpu()), output_upright.detach().cpu(),
                                target_upright.detach().cpu(),
                                str(num_processed_samples)+"_upright_")


            angle = np.random.randint(80, 270)
            image_rot = image.clone()
            image_rot = de_normalize(image_rot)
            image_rot = random_rotate(image=image_rot, angle=angle, interpolation=InterpolationMode.BILINEAR)
            image_rot = mask_corners(image_rot)
            image_rot = normalize(image_rot)

            target_rot = mask_corners(torch.permute(target.clone(), (0,2,3,1)))
            target_rot = random_rotate(image=target_rot, angle=angle, interpolation=InterpolationMode.NEAREST)

            image_rot, target_rot = image_rot.to(device), target_rot.to(device)
            output_rot = model(image_rot)
            output_rot = output_rot["out"]

            confmat_random_rot.update(target_rot.flatten(), output_rot.argmax(1).flatten())

            save_visualizations(de_normalize(image_rot.detach().cpu()), output_rot.detach().cpu(),
                                target_rot.detach().cpu(),
                                str(num_processed_samples)+"_rot_")

            # Find Angle
            _, _, amr_ang = model_amr(image_rot)

            # Compute correction angle
            unrotate_ang = (360 - torch.argmax(amr_ang, 1))
            # Unrotate -based off original image
            unrotate_from_orig = (angle + unrotate_ang) % 360


            image_amr = image.clone()
            image_amr = de_normalize(image_amr)
            image_amr = random_rotate(image=image_amr, angle=unrotate_from_orig, interpolation=InterpolationMode.BILINEAR)
            image_amr = mask_corners(image_amr)
            image_amr = normalize(image_amr)

            target_amr = mask_corners(torch.permute(target.clone(), (0,2,3,1)))
            target_amr = random_rotate(image=target_amr, angle=unrotate_from_orig, interpolation=InterpolationMode.NEAREST)

            image_amr, target_amr = image_amr.to(device), target_amr.to(device)
            output_amr = model(image_amr)
            output_amr = output_amr["out"]

            confmat_amr.update(target_amr.flatten(), output_amr.argmax(1).flatten())

            save_visualizations(de_normalize(image_amr.detach().cpu()), output_amr.detach().cpu(),
                                target_amr.detach().cpu(),
                                str(num_processed_samples)+"_amr_")


            # from PIL import Image
            # to_pil_image(image[0]).show()
            # to_pil_image(target[0]).show()
            # Image.fromarray(de_normalized.type(torch.uint8).numpy()[0]).show()
            # Image.fromarray(target_masked.type(torch.uint8).numpy()[0].squeeze()).show()
            # to_pil_image(image_normalized[0]).show()
            # preprocess image

            num_processed_samples += image.shape[0]

        confmat_upright.reduce_from_all_processes()
        confmat_random_rot.reduce_from_all_processes()
        confmat_amr.reduce_from_all_processes()

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

    return confmat_upright, confmat_random_rot, confmat_amr


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
    confmat_upright, confmat_random_rot, confmat_amr = evaluate(model, data_loader_test, device=device, num_classes=num_classes, preprocess = preprocess,
                       class_to_idx = class_to_idx, model_amr=model_amr)
    print(confmat_upright)
    print(confmat_random_rot)
    print(confmat_amr)




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