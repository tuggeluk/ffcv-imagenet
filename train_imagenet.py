import torch as ch

from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torch.distributed as dist
ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)
#
from torchvision import models

import torchmetrics
import numpy as np
from tqdm import tqdm

import os
import time
import json
from uuid import uuid4
from typing import List
from pathlib import Path
from argparse import ArgumentParser

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf, Checker

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from input_tranformations.mask_corners import MaskCorners
from input_tranformations.rotate_torch import RandomRotate_Torch
from input_tranformations.mask_corners_torch import MaskCorners_Torch
from input_tranformations.own_ops import NormalizeImage, ToTorchImage
import wandb
from rotation_module.angle_classifier_wrapper import AngleClassifierWrapper
from rotation_module.fc_angle_classifier import FcAngleClassifier
from rotation_module.fc2_angle_classifier import Fc2AngleClassifier
from rotation_module.deep_angle_classifier import DeepAngleClassifier
from torchvision.transforms.functional import rotate
from rotation_module.vit_angle_classifier import VitAngleClassifier

class Fastargs_List(Checker):
    def __init__(self, cast_to = str):
        self.cast_to = cast_to

    def check(self, value):
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            value = value.split(",")
            value = [self.cast_to(x.strip()) for x in value]
            return value
        wrap_list = list()
        wrap_list.append(value)
        return wrap_list

    def help(self):
        return "a list"


Section('model', 'model details').params(
    arch=Param(And(str, OneOf(models.__dir__())), default='resnet18'),
    pretrained=Param(int, 'is pretrained? (1/0)', default=0)
)

Section('resolution', 'resolution scheduling').params(
    min_res=Param(int, 'the minimum (starting) resolution', default=160),
    max_res=Param(int, 'the maximum (starting) resolution', default=160),
    end_ramp=Param(int, 'when to stop interpolating resolution', default=0),
    start_ramp=Param(int, 'when to start interpolating resolution', default=0)
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', required=True),
    val_dataset=Param(str, '.dat file to use for validation', required=True),
    num_workers=Param(int, 'The number of workers', required=True),
    in_memory=Param(int, 'does the dataset fit in memory? (1/0)', required=True),
    dataset=Param(OneOf(['ImageNet', 'StanfordCars', 'OxfordPet']), '.dat file to use for training', default='ImageNet')
)

Section('lr', 'lr scheduling').params(
    step_ratio=Param(float, 'learning rate step ratio', default=0.5),
    step_length=Param(int, 'learning rate step length', default=30),
    lr_schedule_type=Param(OneOf(['step', 'cyclic']), default='cyclic'),
    lr=Param(float, 'learning rate', default=0.5),
    lr_peak_epoch=Param(int, 'Epoch at which LR peaks', default=30),
)

Section('logging', 'how to log stuff').params(
    folder=Param(str, 'log location', required=True),
    log_level=Param(int, '0 if only at end 1 otherwise', default=1),
    wandb_dryrun=Param(int, '0 if wanb should be logged', default=1),
    wandb_project=Param(str, 'Name of WandB project', default="ffcv-imagenet"),
    wandb_run=Param(str, 'Name of WandB run', default="default-ffcv-imagenet"),
    wandb_batch_interval=Param(int, 'Interval of batches which are logged to WandB', default=-1)
)

Section('validation', 'Validation parameters stuff').params(
    batch_size=Param(int, 'The batch size for validation', default=512),
    resolution=Param(int, 'final resized validation image size', default=256),
    lr_tta=Param(int, 'should do lr flipping/avging at test time', default=0),
    corner_mask=Param(int, 'should mask corners at test time', default=0),
    random_rotate=Param(int, 'should random rotate at test time', default=0),
    p_flip_upright=Param(float, 'percentage of images to be upright', default=0),
    double_rotate=Param(int, 'rotate everything twice to check if rotation artifacts play a role', default=0),
    double_resize=Param(int, 'resize inbetween rotates only works if double rotate ona late resize are on', default=0),
    pre_flip=Param(int, 'rotate everything twice to check if rotation artifacts play a role', default=0),
    late_resize=Param(int, 'resize after rotation, <0 means do nothing', default=-1),
    load_noise=Param(int, '0: no effect, 1:load random noise instead of images, 2: load blank colors, 3: random lines', default=0),
    interpolation=Param(int, 'interpolation method of rotations, 0:nearest, 1:bilinear, 2: PIL bicubic', default=1)
)

Section('training', 'training hyper param stuff').params(
    eval_only=Param(int, 'eval only?', default=0),
    batch_size=Param(int, 'The batch size', default=512),
    optimizer=Param(And(str, OneOf(['sgd'])), 'The optimizer', default='sgd'),
    momentum=Param(float, 'SGD momentum', default=0.9),
    weight_decay=Param(float, 'weight decay', default=4e-5),
    epochs=Param(int, 'number of epochs', default=30),
    label_smoothing=Param(float, 'label smoothing parameter', default=0.1),
    distributed=Param(int, 'is distributed?', default=0),
    use_blurpool=Param(int, 'use blurpool?', default=1),
    corner_mask=Param(int, 'should mask corners at train time', default=0),
    random_rotate=Param(int, 'should random rotate at train time', default=0),
    checkpoint_interval=Param(int, 'interval of saved checkpoints', default=-1),
    block_rotate=Param(int, 'should the whole tensor be rotated at once', default=0),
    p_flip_upright=Param(float, 'percentage of images to be upright', default=0),
    load_from=Param(str, 'path of pretrained weights', default=""),
    double_rotate=Param(int, 'rotate everything twice to check if rotation artifacts play a role', default=0),
    double_resize=Param(int, 'resize inbetween rotates only works if double rotate and late resize are on', default=0),
    late_resize=Param(int, 'resize after rotation, <0 means do nothing', default=-1),
    load_noise=Param(int, '0: no effect, 1:load random noise instead of images, 2: load blank colors, 3: random lines', default=0),
    interpolation=Param(int, 'interpolation method of rotations, 0:nearest, 1:bilinear, 2: PIL bicubic', default=1)
)

Section('dist', 'distributed training options').params(
    world_size=Param(int, 'number gpus', default=1),
    address=Param(str, 'address', default='localhost'),
    port=Param(str, 'port', default='12252')
)

Section('angleclassifier', 'distributed training options').params(
    attach_upright_classifier=Param(int, 'should an uprightness classifier be added to the model?', default=0),
    attach_ang_classifier=Param(int, 'should an angle classifier be added to the model?', default=0),

    classifier_upright=Param(str, 'which angle classifier should be used', default='deep'),
    classifier_ang=Param(str, 'which angle classifier should be used', default='deep'),

    loss_scope=Param(int, '0: compute loss on img classification, 1: compute loss on angle, 2:combined', default=0),
    freeze_base=Param(int, 'should the base model be frozen?', default=0),
    angle_binsize=Param(Fastargs_List(), 'angle width lumped into one class', default=['lr', 3, 10, 90]),
    prio_class=Param(float, 'should we use regression for the angle', default=1),
    prio_angle=Param(float, 'should we use regression for the angle', default=1),
    flatten=Param(And(str, OneOf(['basic', 'extended'])), 'flatten with avg pool (1,1) or (5,5)', default='extended'),
    shape_class_loss=Param(int, 'perform loss shaping on the angle classifier', default=0)
)

Section('angle_testmode', 'configure how testing performed').params(
    standard=Param(int, 'individually evaluate class/upright/angle predictions', default=1),
    corr_pred=Param(int, 'evaluate angle corrected class prediction', default=0),
)

Section('eval_configs', 'evaluation of basemodels and angleclassifiers').params(
    base_model=Param(int, 'evaluate on a series of base models', default=0),
    angle_class=Param(int, 'evaluate on a series of base angle_classifiers', default=0),
    models_dict=Param(str, 'dir of models to evaluate', default="BaseModels"),
    degree_interval=Param(int, 'Step size of eval angle', default=45),
)



IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

@param('lr.lr')
@param('lr.step_ratio')
@param('lr.step_length')
@param('training.epochs')
def get_step_lr(epoch, lr, step_ratio, step_length, epochs):
    if epoch >= epochs:
        return 0

    num_steps = epoch // step_length
    return step_ratio**num_steps * lr

@param('lr.lr')
@param('training.epochs')
@param('lr.lr_peak_epoch')
def get_cyclic_lr(epoch, lr, epochs, lr_peak_epoch):
    xs = [0, lr_peak_epoch, epochs]
    ys = [1e-4 * lr, lr, 0]
    return np.interp([epoch], xs, ys)[0]

class BlurPoolConv2d(ch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = ch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer('blur_filter', filt)

    def forward(self, x):
        blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
                           groups=self.conv.in_channels, bias=None)
        return self.conv.forward(blurred)

class ImageNetTrainer:
    @param('training.distributed')
    def __init__(self, gpu, distributed):
        self.all_params = get_current_config()
        self.gpu = gpu

        self.uid = str(uuid4())

        if distributed:
            self.setup_distributed()

        self.train_loader = self.create_train_loader()
        self.val_loader = self.create_val_loader()
        self.model, self.scaler = self.create_model_and_scaler()
        # if self.gpu == 0:
        #     print(self.model)
        self.create_optimizer()
        self.initialize_logger()
        

    @param('dist.address')
    @param('dist.port')
    @param('dist.world_size')
    def setup_distributed(self, address, port, world_size):
        os.environ['MASTER_ADDR'] = address
        os.environ['MASTER_PORT'] = port

        dist.init_process_group("nccl", rank=self.gpu, world_size=world_size)
        ch.cuda.set_device(self.gpu)

    def cleanup_distributed(self):
        dist.destroy_process_group()

    @param('lr.lr_schedule_type')
    def get_lr(self, epoch, lr_schedule_type):
        lr_schedules = {
            'cyclic': get_cyclic_lr,
            'step': get_step_lr
        }

        return lr_schedules[lr_schedule_type](epoch)

    # resolution tools
    @param('resolution.min_res')
    @param('resolution.max_res')
    @param('resolution.end_ramp')
    @param('resolution.start_ramp')
    def get_resolution(self, epoch, min_res, max_res, end_ramp, start_ramp):
        assert min_res <= max_res

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res

    @param('training.momentum')
    @param('training.optimizer')
    @param('training.weight_decay')
    @param('training.label_smoothing')
    @param('angleclassifier.attach_ang_classifier')
    def create_optimizer(self, momentum, optimizer, weight_decay,
                         label_smoothing, attach_ang_classifier):
        assert optimizer == 'sgd'

        # Only do weight decay on non-batchnorm parameters
        all_params = list(self.model.named_parameters())
        bn_params = [v for k, v in all_params if ('bn' in k)]
        other_params = [v for k, v in all_params if not ('bn' in k)]
        param_groups = [{
            'params': bn_params,
            'weight_decay': 0.
        }, {
            'params': other_params,
            'weight_decay': weight_decay
        }]

        self.optimizer = ch.optim.SGD(param_groups, lr=1, momentum=momentum)
        self.loss = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        if attach_ang_classifier:
            target_template = 1/(np.abs(np.arange(0, 360, 1)-180)+1)
            target_template[target_template < 0.1] = 0
            target_template = ch.Tensor(target_template/sum(target_template)).to(self.gpu)
            target_template = ch.roll(target_template, -180)
            #self.target_template = target_template.expand(batch_size, target_template.shape[0])
            self.target_template = target_template

    @param('data.train_dataset')
    @param('data.num_workers')
    @param('training.batch_size')
    @param('training.distributed')
    @param('data.in_memory')
    @param('training.corner_mask')
    @param('training.random_rotate')
    @param('training.block_rotate')
    @param('training.p_flip_upright')
    @param('training.double_rotate')
    @param('training.late_resize')
    @param('training.double_resize')
    @param('training.load_noise')
    @param('training.interpolation')
    def create_train_loader(self, train_dataset, num_workers, batch_size, distributed,
                            in_memory, corner_mask, random_rotate, block_rotate, p_flip_upright,
                            double_rotate, late_resize, double_resize, load_noise, interpolation):
        this_device = f'cuda:{self.gpu}'
        train_path = Path(train_dataset)
        assert train_path.is_file()

        res = self.get_resolution(epoch=0)
        self.decoder = RandomResizedCropRGBImageDecoder((res, res))
        image_pipeline: List[Operation] = [
            self.decoder,
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        if random_rotate:
            image_pipeline.insert(3, RandomRotate_Torch(block_rotate, p_flip_upright, double_rotate, late_resize=late_resize,
                                                        double_resize=double_resize, load_noise=load_noise, interpolation=interpolation))

        if corner_mask:
            if random_rotate:
                image_pipeline.insert(4, MaskCorners_Torch())
            else:
                image_pipeline.insert(3, MaskCorners_Torch())


        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device), non_blocking=True)
        ]

        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        loader = Loader(train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        os_cache=in_memory,
                        drop_last=True,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)

        return loader

    @param('data.val_dataset')
    @param('data.num_workers')
    @param('validation.batch_size')
    @param('validation.resolution')
    @param('training.distributed')
    @param('validation.corner_mask')
    @param('validation.random_rotate')
    @param('training.block_rotate')
    @param('validation.p_flip_upright')
    @param('validation.double_rotate')
    @param('validation.pre_flip')
    @param('validation.late_resize')
    @param('validation.double_resize')
    @param('angle_testmode.corr_pred')
    @param('validation.load_noise')
    @param('validation.interpolation')
    def create_val_loader(self, val_dataset, num_workers, batch_size, resolution, distributed, corner_mask,
                          random_rotate, block_rotate, p_flip_upright, double_rotate, pre_flip, late_resize,
                          double_resize, corr_pred, load_noise, interpolation):
        this_device = f'cuda:{self.gpu}'
        val_path = Path(val_dataset)
        assert val_path.is_file()
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        self.val_normalizer = NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            self.val_normalizer
        ]

        if random_rotate:
            self.val_rotate_transform = RandomRotate_Torch(block_rotate, p_flip_upright, double_rotate, pre_flip, corr_pred,
                                                        late_resize=late_resize, double_resize=double_resize, load_noise=load_noise,
                                                        interpolation=interpolation)
            image_pipeline.insert(3, self.val_rotate_transform)

        if corner_mask:
            if random_rotate:
                image_pipeline.insert(4, MaskCorners_Torch())
            else:
                image_pipeline.insert(3, MaskCorners_Torch())

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device),
            non_blocking=True)
        ]

        loader = Loader(val_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=OrderOption.SEQUENTIAL,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)
        return loader

    @param('training.epochs')
    @param('logging.log_level')
    @param('training.checkpoint_interval')
    @param('logging.wandb_dryrun')
    def train(self, epochs, log_level, checkpoint_interval, wandb_dryrun):
        self.eval_and_log({'epoch':(-1)})

        for epoch in range(epochs):
            start_train = time.time()
            res = self.get_resolution(epoch)
            self.decoder.output_size = (res, res)
            train_loss = self.train_loop(epoch)
            train_epoch_time = time.time() - start_train

            if ch.isnan(train_loss) or ch.isinf(train_loss):
                print("Loss is NAN - abort mission")
                break

            if log_level > 0:
                extra_dict = {
                    'train_loss': train_loss.item(),
                    'epoch': epoch,
                    'train_time': train_epoch_time
                }

                self.eval_and_log(extra_dict)

            if checkpoint_interval > 0 and (epoch+1) % checkpoint_interval == 0 and self.gpu == 0:
                ch.save(self.model.state_dict(), self.log_folder / ('epoch_'+str(epoch+1)+'_weights.pt'))

        self.eval_and_log({'epoch':epoch})
        if self.gpu == 0:
            ch.save(self.model.state_dict(), self.log_folder / 'final_weights.pt')
            if not wandb_dryrun:
                self.wandb_run.finish()

    def eval_and_log(self, extra_dict={}):
        start_val = time.time()
        stats = self.val_loop()
        val_time = time.time() - start_val
        if self.gpu == 0:
            log_dict = dict({
                'current_lr': self.optimizer.param_groups[0]['lr'],
                'val_time': val_time
            })
            log_dict = {**log_dict, **stats, **extra_dict}
            self.log(log_dict)

        return stats

    def create_classifier(self, type, num_out, flatten, base_model):
        if type == 'fc':
            ang_class = FcAngleClassifier(base_model, num_out)
        elif type == 'fc2':
            ang_class = Fc2AngleClassifier(base_model, num_out)
        elif type == 'deep':
            ang_class = DeepAngleClassifier(base_model, num_out, flatten=flatten)
        elif type == 'deepx2':
            ang_class = DeepAngleClassifier(base_model, num_out, layers=(2, 2, 2, 2), flatten=flatten)
        elif type == 'deepslant':
            ang_class = DeepAngleClassifier(base_model, num_out, layers=(1, 2, 2, 3), flatten=flatten)
        elif type == 'vit':
            ang_class = VitAngleClassifier(base_model, num_out)
        else:
            raise ValueError("Unknown angleclassifier: " + type)
        return ang_class

    @param('model.arch')
    @param('model.pretrained')
    @param('training.distributed')
    @param('training.use_blurpool')
    @param('training.load_from')
    @param('training.eval_only')
    @param('angleclassifier.freeze_base')
    @param('angleclassifier.loss_scope')
    @param('angleclassifier.flatten')
    @param('angleclassifier.attach_upright_classifier')
    @param('angleclassifier.attach_ang_classifier')
    @param('angleclassifier.classifier_upright')
    @param('angleclassifier.classifier_ang')
    @param('angleclassifier.angle_binsize')
    @param('data.dataset')
    def create_model_and_scaler(self, arch, pretrained, distributed, use_blurpool, load_from, eval_only, freeze_base,
                                loss_scope, flatten, attach_upright_classifier, attach_ang_classifier, classifier_upright,
                                classifier_ang, angle_binsize, dataset):
        scaler = GradScaler()

        if dataset == 'ImageNet':
            n_cls = 1000
        elif dataset == 'StanfordCars':
            n_cls = 196
        elif dataset == 'OxfordPet':
            n_cls = 37

        model = getattr(models, arch)(pretrained=pretrained, num_classes=n_cls)
        def apply_blurpool(mod: ch.nn.Module):
            for (name, child) in mod.named_children():
                if isinstance(child, ch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16): 
                    setattr(mod, name, BlurPoolConv2d(child))
                else: apply_blurpool(child)
        if use_blurpool: apply_blurpool(model)

        model = model.to(memory_format=ch.channels_last)
        # if loss_scope == 1 and not eval_only:
        #     # delete img classifier
        #     model.fc = ch.nn.Identity()

        if attach_upright_classifier:
            upright_class = self.create_classifier(classifier_upright, [2]*len(angle_binsize), flatten, model)
        else:
            upright_class = None

        if attach_ang_classifier:
            ang_class = self.create_classifier(classifier_ang, 360, flatten, model)
        else:
            ang_class = None

        model = AngleClassifierWrapper(model, upright_class, ang_class)

        if not load_from == "":
            from collections import OrderedDict
            state_dict = ch.load(os.path.join(load_from))
            state_dict_renamed = OrderedDict()
            if 'lr_scheduler' in state_dict.keys():
                state_dict = state_dict['model']
                for k, v in state_dict.items():
                    # rename keys
                    kn = "base_model." + k
                    state_dict_renamed[kn] = state_dict[k]

            else:
                for k, v in state_dict.items():
                    #rename keys
                    kn = k[7:]
                    if ((not "up_class" in k) and (not "ang_class" in k)) and (not "base_model" in k):
                        kn = "base_model."+kn
                    state_dict_renamed[kn] = state_dict[k]

            missing, unexpected = model.load_state_dict(state_dict_renamed, strict=False)
            if self.gpu == 0:
                print(" ---------- weights loaded ----------")
                print(" ---------- missing   keys ----------")
                print(missing)
                print(" ---------- unexptected keys ----------")
                print(unexpected)
            # state_dict_noang = ch.load("logs/rn18_base_configs/rn18_88_epochs_mask_norotate.yaml/final_weights.pt")
            # state_dict_renamed_noang = OrderedDict()
            # for k, v in state_dict_noang.items():
            #     #rename keys
            #     kn = k[7:]
            #     if ((not "up_class" in k) and (not "ang_class" in k)) and (not "base_model" in k):
            #         kn = "base_model."+kn
            #     state_dict_renamed_noang[kn] = state_dict_noang[k]
            # for k,v in state_dict_renamed_noang.items():
            #     diff = ch.sum(state_dict_renamed_noang[k] - state_dict_renamed[k])
            #     if diff > 0:
            #         print(k)
        if freeze_base:
            model.freeze_base()
            # self.bn1 = model.base_model.bn1.running_mean

        model = model.to(self.gpu)

        if distributed:
            model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu], find_unused_parameters=True)

        return model, scaler


    @param('angleclassifier.angle_binsize')
    def bin_angles(self, angles, angle_binsize):
        angle_classes = list()
        for b_size in angle_binsize:
            if b_size == 'lr':
                angle_class = (angles > 180)*1
                angle_classes.append(angle_class)
            else:
                offset = int(b_size / 2)
                angle_class = ((angles <= offset) + (angles-360 >= -offset))*1
                angle_classes.append(angle_class)
        return angle_classes

    def bin_sin_cos(self, values):
        binned = ch.sum(ch.unsqueeze(values, -1) < self.bin_tresh[:values.shape[0]], -1)
        return binned

    def prep_angle_target(self, angles, val_mode=False, up_class=True):
        if up_class:
            angles = self.bin_angles(angles)
        else:
            if val_mode:
                angles = angles.type(ch.int64)
            else:
                targets = []
                for i in range(len(angles)):
                    if not val_mode:
                        targets.append(ch.roll(self.target_template, int(angles[i])))
                angles = ch.stack(targets)

        return angles

    def decode_angle(self, output_angle):
        angles = ch.argmax(output_angle, -1)
        return angles

    @param('angleclassifier.angle_binsize')
    @param('angleclassifier.angle_regress')
    def angle_within_binsize(self, output_angle, target_angle, angle_binsize, angle_regress):
        angles = self.decode_angle(output_angle)
        if angle_regress == 2:
            tar_angle = target_angle
        else:
            tar_angle = self.decode_angle(target_angle)

        min_diff = ch.min(ch.stack((ch.abs(tar_angle - angles), ch.abs(ch.abs(tar_angle-angles)-360))), 0)[0]
        min_diff = ch.nan_to_num(min_diff, 359)
        angles_within = (min_diff <= (angle_binsize/2))*1
        return ch.squeeze(angles_within)

    @param('angleclassifier.prio_class')
    @param('angleclassifier.prio_angle')
    def merge_losses(self, loss_class, loss_angle, prio_class, prio_angle):
        loss_tot = prio_class*loss_class+prio_angle*loss_angle
        return loss_tot

    @param('angleclassifier.attach_upright_classifier')
    @param('angleclassifier.attach_ang_classifier')
    @param('angleclassifier.angle_binsize')
    def compute_angle_loss(self, output_up, output_ang, target_up, target_ang, attach_upright_classifier,
                           attach_ang_classifier, angle_binsize):
        tot_loss = 0
        if attach_upright_classifier and target_up is not None:
            for i in range(len(angle_binsize)):
                tot_loss += self.loss(output_up[i], target_up[i])

        if attach_ang_classifier and target_ang is not None:
            tot_loss += self.loss(output_ang, target_ang)

        return tot_loss

    @param('logging.log_level')
    @param('logging.wandb_dryrun')
    @param('logging.wandb_batch_interval')
    @param('angleclassifier.loss_scope')
    @param('angleclassifier.attach_upright_classifier')
    @param('angleclassifier.attach_ang_classifier')
    @param('angleclassifier.shape_class_loss')
    def train_loop(self, epoch, log_level, wandb_dryrun, wandb_batch_interval,
                   loss_scope, attach_upright_classifier, attach_ang_classifier, shape_class_loss):
        model = self.model
        model.train()

        lr_start, lr_end = self.get_lr(epoch), self.get_lr(epoch + 1)
        iters = len(self.train_loader)
        lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

        iterator = tqdm(self.train_loader)
        for ix, (images, target) in enumerate(iterator):

            if isinstance(images, tuple):
                images = tuple(x[:target.shape[0]] for x in images)

                angles = images[1][0]
                if angles is not None:
                    angles = angles[:target.shape[0]]
                pre_norm_imgs = images[1][1]
                if pre_norm_imgs is not None:
                    pre_norm_imgs = pre_norm_imgs[:target.shape[0]]

                target_up = target_ang = None
                if attach_upright_classifier:
                    target_up = self.prep_angle_target(angles, up_class=True)
                if attach_ang_classifier:
                    val_mode = not shape_class_loss
                    target_ang = self.prep_angle_target(angles, up_class=False, val_mode=val_mode)

                images = images[0]

            ### Training start
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lrs[ix]

            self.optimizer.zero_grad(set_to_none=True)
            with autocast():
                output_cls, output_up, output_ang = self.model(images)

                if loss_scope == 0:
                    loss_train = self.loss(output_cls, target)
                elif loss_scope == 1:
                    loss_train = self.compute_angle_loss(output_up, output_ang, target_up, target_ang)
                elif loss_scope == 2:
                    loss_class = self.loss(output_cls, target)
                    loss_angle = self.compute_angle_loss(output_up, output_ang, target_up, target_ang)
                    loss_train = self.merge_losses(loss_class, loss_angle)

            #print("loss-cls:" + str(self.loss(output_cls, target)))
            self.scaler.scale(loss_train).backward()

            self.scaler.unscale_(self.optimizer)
            ch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            ### Training end
            # print(ch.sum(self.fc_weight-self.model.module.base_model.fc.weight))
            # print(ch.sum(self.conv1-self.model.module.base_model.layer1[0].conv1.weight))
            # print(ch.sum(self.bn1 - self.model.module.base_model.bn1.running_mean.cpu()))

            ### Logging start
            if log_level > 0:

                group_lrs = []
                for _, group in enumerate(self.optimizer.param_groups):
                    group_lrs.append(f'{group["lr"]:.5f}')

                names = ['epoch', 'iter', 'lrs']
                values = [epoch, ix, group_lrs]

                names += ['loss_train']
                values += [f'{loss_train.item():.3f}']

                if loss_scope==2:
                    names += ['loss_class']
                    values += [f'{loss_class.item():.3f}']
                    names += ['loss_angle']
                    values += [f'{loss_angle.item():.3f}']

                msg = ', '.join(f'{n}={v}' for n, v in zip(names, values))
                iterator.set_description(msg)
                if self.gpu == 0:
                    if not wandb_dryrun:
                        if wandb_batch_interval > 0 and ix % wandb_batch_interval == 0:
                            wandb_log_dict = {}
                            for name, value in  zip(names,values):
                                if isinstance(value, list):
                                    for i in range(len(value)):
                                        wandb_log_dict[name+'_'+str(i)] = float(value[i])
                                else:
                                    wandb_log_dict[name] = float(value)

                            wandb.log(wandb_log_dict)
            ### Logging end

        return loss_train


    @param('angleclassifier.loss_scope')
    @param('angleclassifier.attach_upright_classifier')
    @param('angleclassifier.attach_ang_classifier')
    @param('angleclassifier.angle_binsize')
    @param('angle_testmode.standard')
    @param('angle_testmode.corr_pred')
    def val_loop(self, loss_scope, attach_upright_classifier, attach_ang_classifier, angle_binsize, standard, corr_pred):
        model = self.model
        model.eval()

        errs_mod_90 = None

        with ch.no_grad():
            with autocast():
                for images, target in tqdm(self.val_loader):
                    target_up = target_ang = None

                    if isinstance(images, tuple):
                        images = tuple(x[:target.shape[0]] for x in images)

                        angles = images[1][0]
                        if angles is not None:
                            angles = angles[:target.shape[0]]
                        orig_imgs = images[1][1]
                        if orig_imgs is not None:
                            orig_imgs = orig_imgs[:target.shape[0]]

                        if attach_upright_classifier:
                            target_up = self.prep_angle_target(angles, up_class=True, val_mode=True)
                        if attach_ang_classifier:
                            target_ang = self.prep_angle_target(angles, up_class=False, val_mode=True)

                        images = images[0]

                    # pil_mode = False
                    # if pil_mode:
                    #     dtype = images.dtype
                    #     images = orig_imgs.clone()
                    #     images = images.permute(0,3,1,2)
                    #     images = self.rotate_images(images, angles, PIL_mode = True)
                    #     images = self.mask_corners(images)
                    #     images = self.norm_images(images, dtype)

                    # target_ang = ch.tensor([0, 30, 60, 90, 120, 150, 180, 210, 270,
                    #                        300, 330, 0, 180, 210, 240, 270, 300, 330]).to(target_ang.device)


                    output_cls, output_up, output_ang = self.model(images)

                    if standard:
                        #if loss_scope == 0 or loss_scope == 2:
                        for k in ['top_1_class', 'top_5_class']:
                            self.val_meters[k](output_cls, target)

                        loss_val = self.loss(output_cls, target)
                        self.val_meters['loss_class'](loss_val)

                        #print("val loss clss: " + str(self.loss(output_cls, target)))

                        if loss_scope == 1 or loss_scope == 2:
                            if attach_upright_classifier and target_up is not None:
                                for i_bsize, b_size in enumerate(angle_binsize):
                                    self.val_meters['top_1_angle_upright_' + str(b_size) + "_binsize"](output_up[i_bsize], target_up[i_bsize])

                            if attach_ang_classifier and target_ang is not None:
                                for k in ['top_1_angle', 'top_5_angle']:
                                    self.val_meters[k](output_ang, target_ang)

                                # dummy_target = ch.ones(target_ang.shape[0]).type(ch.int).to(self.gpu)
                                # angle_within = self.angle_within_binsize(output_ang, target_ang)
                                # self.val_meters['top_1_within_binsize'](angle_within, dummy_target)
                            if attach_upright_classifier or attach_ang_classifier:
                                loss_ang = self.compute_angle_loss(output_up, output_ang, target_up, target_ang)
                                self.val_meters['loss_angle'](loss_ang)

                    if corr_pred:
                        assert attach_ang_classifier
                        # from PIL import Image
                        # Image.fromarray(orig_imgs[1].cpu().numpy()).show()
                        # Image.fromarray(corr_imgs[1].cpu().numpy().transpose(1,2,0)).show()

                        corr_imgs = orig_imgs.clone()
                        corr_imgs = corr_imgs.permute(0,3,1,2)
                        unrotate_ang = (360 - ch.argmax(output_ang, 1))
                        # Unrotate -based off original image
                        unrotate_from_orig = (angles + unrotate_ang)%360

                        corr_imgs = self.rotate_images(corr_imgs, unrotate_from_orig)
                        corr_imgs = self.mask_corners(corr_imgs)

                        # # create sanity-check image
                        # from PIL import Image
                        # orig_imgs_san = orig_imgs.clone().cpu()
                        # corr_imgs_san = corr_imgs.clone().cpu()
                        # orig_imgs_san = self.mask_corners(orig_imgs_san.permute(0, 3, 1, 2))
                        # orig_imgs_san = orig_imgs_san.numpy()
                        # corr_imgs_san = corr_imgs_san.numpy()
                        # target_san = target.clone().cpu().numpy()
                        #
                        # for i in np.unique(target_san):
                        #     inps = np.concatenate(orig_imgs_san[target_san == i].transpose(0, 2, 3, 1), axis=1)
                        #     corrs = np.concatenate(corr_imgs_san[target_san == i].transpose(0, 2, 3, 1), axis=1)
                        #
                        #     Image.fromarray(np.concatenate([inps, corrs], axis=0).astype(np.uint8)).show()
                        #     Image.fromarray(np.concatenate([inps, corrs], axis=0).astype(np.uint8)).save(str(i)+"sanity.png")
                        # # end sanity check image


                        corr_imgs = self.norm_images(corr_imgs, images.dtype)
                        output_cls_corr_pred, ouput_up_corr_pred, ouput_ang_corr_pred = self.model(corr_imgs)


                        refine_predictions = False
                        if refine_predictions:
                            refine_angles = [90, 180, 270]
                            output_cls_corr_preds = [output_cls_corr_pred]
                            ouput_up_corr_preds = [ouput_up_corr_pred]
                            ouput_ang_corr_preds = [ouput_ang_corr_pred]
                            uprights = [ch.argmax(ouput_up_corr_pred[1],1)]

                            for i in range(len(refine_angles)):
                                refine_imgs = orig_imgs.clone()
                                refine_imgs = refine_imgs.permute(0, 3, 1, 2)

                                refine_angle = unrotate_from_orig+refine_angles[i]%360

                                refine_imgs = self.rotate_images(refine_imgs, refine_angle)
                                refine_imgs = self.mask_corners(refine_imgs)
                                refine_imgs = self.norm_images(refine_imgs, images.dtype)

                                #not_upright = unrotate_from_orig[ch.argmax(ouput_up_corr_preds[i][1],1)==0]
                                output_cls_corr_pred, ouput_up_corr_pred, ouput_ang_corr_pred = self.model(refine_imgs)
                                output_cls_corr_preds.append(output_cls_corr_pred)
                                ouput_up_corr_preds.append(ouput_up_corr_pred)
                                ouput_ang_corr_preds.append(ouput_ang_corr_pred)
                                uprights.append(ch.argmax(ouput_up_corr_pred[1],1))


                            cls_preds = ch.stack(output_cls_corr_preds)

                            # uprights = ch.stack(uprights)
                            # no_upright = (sum(uprights,0)==0)
                            # uprights[0, no_upright] = 1
                            # uprights = ch.unsqueeze(uprights, -1)
                            # cls_preds = uprights * cls_preds

                            output_cls_corr_pred = ch.sum(cls_preds,0)

                        # mod_90 = (angles + unrotate_ang) % 90
                        # if errs_mod_90 is None:
                        #     errs_mod_90 = ch.min(mod_90, ch.abs(mod_90-90))
                        # else:
                        #     errs_mod_90 = ch.cat((errs_mod_90, ch.min(mod_90, ch.abs(mod_90-90))))

                        for k in ['top_1_class_corr_pred', 'top_5_class_corr_pred']:
                            self.val_meters[k](output_cls_corr_pred, target)


        # stats = dict()
        # for k, m in self.val_meters.items():
        #     print(k)
        #     stats[k] = m.compute().item()
        #print(ch.bincount(errs_mod_90))

        stats = {k: m.compute().item() for k, m in self.val_meters.items()}
        [meter.reset() for meter in self.val_meters.values()]
        return stats

    def norm_images(self, pre_normed, dtype):
        table = ((ch.arange(256)[:, None] - IMAGENET_MEAN[None, :]) / IMAGENET_STD[None, :])
        table = table.type(dtype)
        table = table.to(self.gpu)
        normed = pre_normed.clone().type(dtype)
        for i in range(3):
            normed[:, i, :, :] = table[pre_normed[:, i, :, :].view(-1).type(ch.long), i].reshape(
                pre_normed[:, i, :, :].shape)
        return normed

    def rotate_images(self, images, angles, PIL_mode = False):
        if not PIL_mode:
            for i in range(len(angles)):
                images[i] = rotate(images[i], int(angles[i]))
        else:
            from PIL import Image
            images_np = np.array(images.cpu()).transpose(0,2,3,1)
            for i in range(len(angles)):
                images_np[i] = np.array(Image.fromarray(images_np[i]).rotate(int(angles[i]), Image.Resampling.BICUBIC))
                #Image.fromarray(images_np[i]).rotate(30, Image.Resampling.BICUBIC).show()
            images_np = images_np.transpose(0,3,1,2)
            images = ch.Tensor(images_np).to(self.gpu).type(images.dtype)
        return images

    def mask_corners(self, images):
        images = images.permute(0, 2, 3, 1)
        x = np.arange(0, images[0].shape[0], 1) - np.floor(images[0].shape[0] / 2)
        y = np.arange(0, images[0].shape[1], 1) - np.floor(images[0].shape[1] / 2)
        xx, yy = np.meshgrid(x, y)
        mask = (np.sqrt((xx * xx) + (yy * yy)) - images[0].shape[0] / 2) > -3
        images[:, mask, :] = 255
        images = images.permute(0, 3, 1, 2)
        return images

    @param('logging.folder')
    @param('logging.wandb_dryrun')
    @param('logging.wandb_project')
    @param('logging.wandb_run')
    @param('angleclassifier.loss_scope')
    @param('angleclassifier.attach_upright_classifier')
    @param('angleclassifier.attach_ang_classifier')
    @param('angleclassifier.angle_binsize')
    @param('angle_testmode.corr_pred')
    def initialize_logger(self, folder, wandb_dryrun, wandb_project, wandb_run, loss_scope, attach_upright_classifier,
                          attach_ang_classifier, angle_binsize, corr_pred):
        self.val_meters = {}

        if loss_scope == 0 or loss_scope == 2 or True:
            self.val_meters['top_1_class'] = torchmetrics.Accuracy(compute_on_step=False).to(self.gpu)
            self.val_meters['top_5_class'] = torchmetrics.Accuracy(compute_on_step=False, top_k=5).to(self.gpu)
            self.val_meters['loss_class'] = MeanScalarMetric(compute_on_step=False).to(self.gpu)
        if loss_scope == 1 or loss_scope == 2:
            if attach_upright_classifier:
                for b_size in angle_binsize:
                    self.val_meters['top_1_angle_upright_'+str(b_size)+"_binsize"] = torchmetrics.Accuracy(compute_on_step=False).to(self.gpu)

            if attach_ang_classifier:
                self.val_meters['top_1_angle'] = torchmetrics.Accuracy(compute_on_step=False).to(self.gpu)
                self.val_meters['top_5_angle'] = torchmetrics.Accuracy(compute_on_step=False, top_k=5).to(self.gpu)

                # self.val_meters['top_1_within_binsize'] = torchmetrics.Accuracy(compute_on_step=False).to(self.gpu)

            self.val_meters['loss_angle'] = MeanScalarMetric(compute_on_step=False).to(self.gpu)

        if corr_pred:
            self.val_meters['top_1_class_corr_pred'] = torchmetrics.Accuracy(compute_on_step=False).to(self.gpu)
            self.val_meters['top_5_class_corr_pred'] = torchmetrics.Accuracy(compute_on_step=False, top_k=5).to(self.gpu)


        if self.gpu == 0:
            #folder = (Path(folder) / str(self.uid)).absolute()
            folder = (Path(folder)).absolute()
            folder.mkdir(parents=True, exist_ok=True)

            self.log_folder = folder
            self.start_time = time.time()

            print(f'=> Logging in {self.log_folder}')
            params = {
                '.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()
            }

            with open(folder / 'params.json', 'w+') as handle:
                json.dump(params, handle)

            if not wandb_dryrun:
                self.wandb_run = wandb.init(project=wandb_project, name=wandb_run, reinit=True)
                wandb_config_dict = {}
                for k in self.all_params.content.keys():
                    wandb_config_dict[str(k)] = self.all_params.content[k]
                wandb.config.update(wandb_config_dict)

    @param('logging.wandb_dryrun')
    def log(self, content, wandb_dryrun):
        print(f'=> Log: {content}')
        if self.gpu != 0: return
        cur_time = time.time()
        with open(self.log_folder / 'log', 'a+') as fd:
            fd.write(json.dumps({
                'timestamp': cur_time,
                'relative_time': cur_time - self.start_time,
                **content
            }) + '\n')
            fd.flush()
        if not wandb_dryrun:
            removes = ['current_lr','train_loss']
            for rem in removes:
                if rem in content.keys():
                    del content[rem]

            wandb.log(content)

    def evaluate_base_models(self):
        # set up csv folder
        # go through epochs, compute top1,top5 on rotate and no rotate
        # final epoch also compute top1,top5 per angle
        return None

    @param('eval_configs.degree_interval')
    def evaluate_angle_class(self, degree_interval):
        # set up csv folder
        # final epoch, compute top1, top5
        for i in np.arange(0, 360, degree_interval):
            self.val_rotate_transform.set_angle_config(i)
            self.eval_and_log()
        return None

    def make_clear_csv_folder(self):
        return None


    @classmethod
    @param('training.distributed')
    @param('dist.world_size')
    def launch_from_args(cls, distributed, world_size):
        if distributed:
            ch.multiprocessing.spawn(cls._exec_wrapper, nprocs=world_size, join=True)
        else:
            cls.exec(0)

    @classmethod
    def _exec_wrapper(cls, *args, **kwargs):
        make_config(quiet=True)
        cls.exec(*args, **kwargs)

    @classmethod
    @param('training.distributed')
    @param('training.eval_only')
    @param('eval_configs.base_model')
    @param('eval_configs.angle_class')
    def exec(cls, gpu, distributed, eval_only, base_model, angle_class):
        trainer = cls(gpu=gpu)
        if eval_only:
            if base_model:
                trainer.evaluate_base_models()
            elif angle_class:
                trainer.evaluate_angle_class()
            else:
                trainer.eval_and_log()
        else:
            trainer.train()

        if distributed:
            trainer.cleanup_distributed()

# Utils
class MeanScalarMetric(torchmetrics.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_state('sum', default=ch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('count', default=ch.tensor(0), dist_reduce_fx='sum')

    def update(self, sample: ch.Tensor):
        self.sum += sample.sum()
        self.count += sample.numel()

    def compute(self):
        return self.sum.float() / self.count

# Running
def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()

if __name__ == "__main__":
    make_config()
    ImageNetTrainer.launch_from_args()
