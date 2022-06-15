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
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from input_tranformations.mask_corners import MaskCorners
from input_tranformations.rotate import RandomRotate
from input_tranformations.rotate_torch import RandomRotate_Torch
import wandb
from rotation_module.angle_classifier_wrapper import AngleClassifierWrapper


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
    in_memory=Param(int, 'does the dataset fit in memory? (1/0)', required=True)
)

Section('lr', 'lr scheduling').params(
    step_ratio=Param(Fastargs_List(float), 'learning rate step ratio', default=0.1),
    step_length=Param(Fastargs_List(int), 'learning rate step length', default=30),
    lr_schedule_type=Param(Fastargs_List(str), default='cyclic'),
    lr=Param(Fastargs_List(float), 'learning rate', default=0.6),
    lr_peak_epoch=Param(Fastargs_List(int), 'Epoch at which LR peaks', default=2),
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
    resolution=Param(int, 'final resized validation image size', default=224),
    lr_tta=Param(int, 'should do lr flipping/avging at test time', default=1),
    corner_mask=Param(int, 'should mask corners at test time', default=0),
    random_rotate=Param(int, 'should random rotate at test time', default=0)
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
    use_blurpool=Param(int, 'use blurpool?', default=0),
    corner_mask=Param(int, 'should mask corners at train time', default=0),
    random_rotate=Param(int, 'should random rotate at train time', default=0),
    checkpoint_interval=Param(int, 'interval of saved checkpoints', default=-1),
    block_rotate=Param(int, 'should the whole tensor be rotated at once', default=0),
    p_flip_upright=Param(float, 'percentage of images to be upright', default=0.5),
)

Section('dist', 'distributed training options').params(
    world_size=Param(int, 'number gpus', default=1),
    address=Param(str, 'address', default='localhost'),
    port=Param(str, 'port', default='12355')
)

Section('angleclassifier', 'distributed training options').params(
    attach_classifier=Param(int, 'should an angle classifier be added to the model?', default=1),
    train_base=Param(int, 'should the image classifier also be trained', default=1),
    optimizer_scopes=Param(Fastargs_List(str), 'for which scopes should an optimizer be built', default='base'),
    optimizer_targets=Param(Fastargs_List(int), 'what output is targeted images or angles', default='0'),
    angle_binary=Param(int, 'binary "uprightness" classification or angle regression', default=1),
    angle_binsize=Param(int, 'span of angles lumped into one class', default=4)

)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

@param('lr.lr')
@param('lr.step_ratio')
@param('lr.step_length')
@param('training.epochs')
def get_step_lr(epoch, opt_ind, lr, step_ratio, step_length, epochs):
    if epoch >= epochs:
        return 0
    lr = select_opt_ind(opt_ind, lr)
    step_ratio = select_opt_ind(opt_ind, step_ratio)
    step_length = select_opt_ind(opt_ind, step_length)

    num_steps = epoch // step_length
    return step_ratio**num_steps * lr


def select_opt_ind(opt_ind,configs):
    if len(configs) == 1:
        return configs[0]
    else:
        return configs[opt_ind]

@param('lr.lr')
@param('training.epochs')
@param('lr.lr_peak_epoch')
def get_cyclic_lr(epoch, opt_ind, lr, epochs, lr_peak_epoch):
    lr = select_opt_ind(opt_ind, lr)
    lr_peak_epoch = select_opt_ind(opt_ind, lr_peak_epoch)
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
    @param('angleclassifier.optimizer_scopes')
    def __init__(self, gpu, distributed, optimizer_scopes):
        self.all_params = get_current_config()
        self.gpu = gpu

        self.uid = str(uuid4())

        if distributed:
            self.setup_distributed()

        self.train_loader = self.create_train_loader()
        self.val_loader = self.create_val_loader()
        self.model, self.scaler = self.create_model_and_scaler()

        self.optimizers = []
        self.losses = []
        for scope in optimizer_scopes:
            optimizer, loss = self.create_optimizer(scope_filter=scope)
            self.optimizers.append(optimizer)
            self.losses.append(loss)

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
    def get_lr(self, epoch, opt_ind, lr_schedule_type):
        lr_schedules = {
            'cyclic': get_cyclic_lr,
            'step': get_step_lr
        }

        return lr_schedules[select_opt_ind(opt_ind, lr_schedule_type)](epoch, opt_ind)

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
    def create_optimizer(self, momentum, optimizer, weight_decay,
                         label_smoothing, scope_filter = 'all'):
        assert optimizer == 'sgd'
        assert scope_filter in ['all', 'base', 'angle']

        # Only do weight decay on non-batchnorm parameters
        all_params = list(self.model.named_parameters())

        if scope_filter == 'base':
            all_params = {(k,v) for k,v in all_params if ('base_model' in k)}
        if scope_filter == 'angle':
            all_params = {(k,v) for k,v in all_params if ('base_model' not in k)}


        bn_params = [v for k, v in all_params if ('bn' in k)]
        other_params = [v for k, v in all_params if not ('bn' in k)]
        param_groups = [{
            'params': bn_params,
            'weight_decay': 0.
        }, {
            'params': other_params,
            'weight_decay': weight_decay
        }]

        optimizer = ch.optim.SGD(param_groups, lr=1, momentum=momentum)
        loss = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        return optimizer, loss

    @param('data.train_dataset')
    @param('data.num_workers')
    @param('training.batch_size')
    @param('training.distributed')
    @param('data.in_memory')
    @param('training.corner_mask')
    @param('training.random_rotate')
    @param('training.block_rotate')
    @param('training.p_flip_upright')
    def create_train_loader(self, train_dataset, num_workers, batch_size,
                            distributed, in_memory, corner_mask, random_rotate, block_rotate, p_flip_upright):
        this_device = f'cuda:{self.gpu}'
        train_path = Path(train_dataset)
        assert train_path.is_file()

        res = self.get_resolution(epoch=0)
        self.decoder = RandomResizedCropRGBImageDecoder((res, res))
        image_pipeline: List[Operation] = [
            self.decoder,
            #RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        # if random_rotate:
        #     image_pipeline.insert(3, RandomRotate_Torch(block_rotate))

        if random_rotate:
            image_pipeline.append(RandomRotate_Torch(block_rotate, p_flip_upright))

        if corner_mask:
            image_pipeline.insert(1, MaskCorners())

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
    def create_val_loader(self, val_dataset, num_workers, batch_size,
                          resolution, distributed, corner_mask, random_rotate, block_rotate):
        this_device = f'cuda:{self.gpu}'
        val_path = Path(val_dataset)
        assert val_path.is_file()
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        # if random_rotate:
        #     image_pipeline.insert(3, RandomRotate_Torch(block_rotate))

        if random_rotate:
            image_pipeline.append(RandomRotate_Torch(block_rotate))

        if corner_mask:
            image_pipeline.insert(1, MaskCorners())

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
        for epoch in range(epochs):
            start_train = time.time()
            res = self.get_resolution(epoch)
            self.decoder.output_size = (res, res)
            train_loss = self.train_loop(epoch)
            train_epoch_time = time.time() - start_train

            if log_level > 0:
                extra_dict = {
                    'train_loss': train_loss,
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
            self.log(dict({
                'current_lr': self.base_optimizer.param_groups[0]['lr'],
                'top_1': stats['top_1'],
                'top_5': stats['top_5'],
                'val_time': val_time
            }, **extra_dict))

        return stats

    @param('model.arch')
    @param('model.pretrained')
    @param('training.distributed')
    @param('training.use_blurpool')
    @param('angleclassifier.attach_classifier')
    def create_model_and_scaler(self, arch, pretrained, distributed, use_blurpool, attach_classifier):
        scaler = GradScaler()
        model = getattr(models, arch)(pretrained=pretrained)
        def apply_blurpool(mod: ch.nn.Module):
            for (name, child) in mod.named_children():
                if isinstance(child, ch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16): 
                    setattr(mod, name, BlurPoolConv2d(child))
                else: apply_blurpool(child)
        if use_blurpool: apply_blurpool(model)

        model = model.to(memory_format=ch.channels_last)
        if attach_classifier:
            model = AngleClassifierWrapper(model)

        model = model.to(self.gpu)

        if distributed:
            model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])

        return model, scaler



    @param('angleclassifier.angle_binsize')
    def bin_angles(self, angles, angle_binsize):

        offset = int(angle_binsize / 2)
        angle_class = ((angles < offset) + (angles-360 > -offset))*1

        return angle_class


    @param('logging.log_level')
    @param('logging.wandb_dryrun')
    @param('logging.wandb_batch_interval')
    @param('angleclassifier.optimizer_targets')
    @param('angleclassifier.angle_binary')
    def train_loop(self, epoch, log_level, wandb_dryrun, wandb_batch_interval, optimizer_targets, angle_binary):
        model = self.model
        model.train()

        lrs = []
        for i in range(len(self.optimizers)):
            lr_start, lr_end = self.get_lr(epoch, i), self.get_lr(epoch + 1, i)
            iters = len(self.train_loader)
            lrs.append(np.interp(np.arange(iters), [0, iters], [lr_start, lr_end]))

        iterator = tqdm(self.train_loader)
        for ix, (images, target) in enumerate(iterator):
            target = [target]
            if isinstance(images, tuple):
                if angle_binary:
                    target.append(self.bin_angles(images[1]))
                else:
                    target.append(images[1])
                images = images[0]


            ### Training start
            for i in range(len(self.optimizers)):
                for param_group in self.optimizers[i].param_groups:
                    param_group['lr'] = lrs[i][ix]
                self.optimizers[i].zero_grad(set_to_none=True)


            with autocast():
                output = self.model(images)

            sub_losses = []

            for i in range(len(self.optimizers)):
                if i == 1:
                    tar = (ch.rand(target[1].shape)>0.5)*1
                    tar = tar.to(images.device)
                    loss_train = self.losses[i](output[optimizer_targets[i]], tar)
                else:
                    loss_train = self.losses[i](output[optimizer_targets[i]], target[optimizer_targets[i]])

                if i == len(self.optimizers)-1:
                    retain_graph = False
                else:
                    retain_graph = True

                self.scaler.scale(loss_train).backward(retain_graph=retain_graph)
                self.scaler.step(self.optimizers[i])
                sub_losses.append(loss_train)


            self.scaler.update()
            ### Training end

            ### Logging start
            #if log_level > 0:
            if False:

                names = ['epoch', 'iter']
                values = [epoch, ix]

                for i, opt in enumerate(self.optimizers):
                    for ii, group in enumerate(opt.param_groups):
                        names.append('lr_o'+str(i)+'_g'+str(ii))
                        values.append(f'{group["lr"]:.5f}')

                for i, loss in enumerate(sub_losses):
                    loss.detach()
                    names.append('loss_train_'+str(i))
                    values.append(f'{loss.item():.3f}')

                msg = ', '.join(f'{n}={v}' for n, v in zip(names, values))
                iterator.set_description(msg)
                if self.gpu == 0:
                    if not wandb_dryrun:
                        if wandb_batch_interval > 0 and ix % wandb_batch_interval == 0:
                            wandb_log_dict = {}
                            for name, value in  zip(names, values):
                                if isinstance(value, list):
                                    for i in range(len(value)):
                                        wandb_log_dict[name+'_'+str(i)] = float(value[i])
                                else:
                                    wandb_log_dict[name] = float(value)

                            wandb.log(wandb_log_dict)


            ### Logging end

    @param('validation.lr_tta')
    def val_loop(self, lr_tta):
        model = self.model
        model.eval()

        with ch.no_grad():
            with autocast():
                for images, target in tqdm(self.val_loader):
                    output = self.model(images)
                    if lr_tta:
                        output += self.model(ch.flip(images, dims=[3]))

                    for k in ['top_1', 'top_5']:
                        self.val_meters[k](output, target)

                    loss_val = self.base_loss(output, target)
                    self.val_meters['loss'](loss_val)

        stats = {k: m.compute().item() for k, m in self.val_meters.items()}
        [meter.reset() for meter in self.val_meters.values()]
        return stats

    @param('logging.folder')
    @param('logging.wandb_dryrun')
    @param('logging.wandb_project')
    @param('logging.wandb_run')
    def initialize_logger(self, folder, wandb_dryrun, wandb_project, wandb_run):
        self.val_meters = {
            'top_1': torchmetrics.Accuracy(compute_on_step=False).to(self.gpu),
            'top_5': torchmetrics.Accuracy(compute_on_step=False, top_k=5).to(self.gpu),
            'loss': MeanScalarMetric(compute_on_step=False).to(self.gpu)
        }

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
    def exec(cls, gpu, distributed, eval_only):
        trainer = cls(gpu=gpu)
        if eval_only:
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
