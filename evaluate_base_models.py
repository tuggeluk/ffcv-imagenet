import torch as ch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torch.distributed as dist
from collections import OrderedDict
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
from fastargs.validation import And, OneOf

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, RandomHorizontalFlip
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from input_tranformations.mask_corners import MaskCorners
from input_tranformations.own_ops import NormalizeImage, ToTorchImage
from input_tranformations.rotate_torch import RandomRotate_Torch
import wandb


Section('model', 'model details').params(
    arch=Param(And(str, OneOf(models.__dir__())), default='resnet18'),
    pretrained=Param(int, 'is pretrained? (1/0)', default=0),
    nr_cl=Param(int, 'number of classes', default=1000),
)

Section('data', 'data related stuff').params(
    val_dataset=Param(str, '.dat file to use for validation', default="/tmp"),
    num_workers=Param(int, 'The number of workers', default=8),
    in_memory=Param(int, 'does the dataset fit in memory? (1/0)', default=1)
)

Section('logging', 'how to log stuff').params(
    log_folder=Param(str, 'log location', required=True),
    log_level=Param(int, '0 if only at end 1 otherwise', default=1),
    wandb_dryrun=Param(int, '0 if wanb should be logged', default=1),
    wandb_project=Param(str, 'Name of WandB project', default="ffcv-imagenet"),
    wandb_run=Param(str, 'Name of WandB run', default="default-ffcv-imagenet"),
    wandb_batch_interval=Param(int, 'Interval of batches which are logged to WandB', default=-1)
)

Section('validation', 'Validation parameters stuff').params(
    batch_size=Param(int, 'The batch size for validation', default=256),
    resolution=Param(int, 'final resized validation image size', default=224),
    corner_mask=Param(int, 'should mask corners at test time', default=0),
    random_rotate=Param(int, 'should random rotate at test time', default=0),
    use_blurpool=Param(int, 'use blurpool?', default=0)
)

Section('multi_validate', 'Multi valdiation parameters').params(
    models_folder=Param(str, 'Destination of pretrained Models', required=True),
    random_runs=Param(int, 'Number of runs with random angles to avg over', default=1),
    degree_interval=Param(int, 'Step size evaluation angle', default=45),
    add_nonrotate_run=Param(int, 'Additionally evaluate without rotating', default=0),
    final_chkp_only=Param(int, 'only eval latest checkpoint', default=0)
)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256


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

class MultiModelEvaluator:
    def __init__(self, gpu):
        self.all_params = get_current_config()
        self.gpu = gpu

        self.uid = str(uuid4())

        os.environ['MASTER_ADDR'] = "localhost"
        os.environ['MASTER_PORT'] = "12355"

        dist.init_process_group("nccl", rank=self.gpu, world_size=1)
        ch.cuda.set_device(self.gpu)

        self.val_loader = self.create_val_loader()
        self.model, self.scaler = self.create_model_and_scaler()
        self.loss = ch.nn.CrossEntropyLoss(label_smoothing=0.1)



    @param('data.val_dataset')
    @param('data.num_workers')
    @param('validation.batch_size')
    @param('validation.resolution')
    @param('validation.corner_mask')
    @param('validation.random_rotate')
    def create_val_loader(self, val_dataset, num_workers, batch_size,
                          resolution, corner_mask, random_rotate):
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

        self.rotate_transform = RandomRotate_Torch()
        if random_rotate:
            image_pipeline.insert(3, self.rotate_transform)

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
                        })
        return loader


    def eval_and_log(self, extra_dict={}):
        start_val = time.time()
        stats = self.val_loop()
        val_time = time.time() - start_val
        if self.gpu == 0:
            self.log(dict({
                'current_lr': self.optimizer.param_groups[0]['lr'],
                'top_1': stats['top_1'],
                'top_5': stats['top_5'],
                'val_time': val_time
            }, **extra_dict))
        return stats

    @param('model.arch')
    @param('model.pretrained')
    @param('validation.use_blurpool')
    @param('model.nr_cl')
    def create_model_and_scaler(self, arch, pretrained, use_blurpool, nr_cl):
        scaler = GradScaler()
        model = getattr(models, arch)(pretrained=pretrained, num_classes=nr_cl)
        def apply_blurpool(mod: ch.nn.Module):
            for (name, child) in mod.named_children():
                if isinstance(child, ch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16): 
                    setattr(mod, name, BlurPoolConv2d(child))
                else: apply_blurpool(child)
        if use_blurpool: apply_blurpool(model)

        model = model.to(memory_format=ch.channels_last)
        model = model.to(self.gpu)

        model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])

        return model, scaler


    def val_loop(self):
        model = self.model
        model.eval()

        with ch.no_grad():
            with autocast():
                for images, target in tqdm(self.val_loader):
                    if isinstance(images, tuple):
                        images = tuple(x[:target.shape[0]] for x in images)
                        angles = images[1][0]
                        images = images[0]

                    output = self.model(images)

                    for k in ['top_1', 'top_5']:
                        self.val_meters[k](output, target)

                    loss_val = self.loss(output, target)
                    self.val_meters['loss'](loss_val)

        stats = {k: m.compute().item() for k, m in self.val_meters.items()}
        [meter.reset() for meter in self.val_meters.values()]
        return stats


    @param('logging.wandb_dryrun')
    def initialize_logger(self, folder, wandb_project, wandb_run, wandb_dryrun):
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
                # check if run exists

                self.wandb_run = wandb.init(project=wandb_project, name=wandb_run, reinit=True)
                wandb_config_dict = {}
                for k in self.all_params.content.keys():
                    wandb_config_dict[str(k)] = self.all_params.content[k]
                wandb.config.update(wandb_config_dict)
        return 1

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
            removes = ['current_lr', 'train_loss']
            for rem in removes:
                if rem in content.keys():
                    del content[rem]

            wandb.log(content)

    def collect_checkpoints(self, path):
        checkpoints_dict = {}
        for file in os.listdir(path):
            if file[:6] == 'epoch_':
                checkpoints_dict[int(file.split("_")[1])] = file

        return checkpoints_dict

    def add_dicts(self, collected_stats, new_stats):
        if collected_stats is None:
            collected_stats = {'top_1': [], 'top_5': [], 'loss': []}
        for k in collected_stats.keys():
            collected_stats[k].append(new_stats[k])
        return collected_stats

    @param('multi_validate.random_runs')
    @param('multi_validate.degree_interval')
    def evaluate_checkpoint(self, name, path, last_entry, rotate, random_runs, degree_interval):
        if not last_entry:
            return

        # load model weights
        state_dict = ch.load(os.path.join(path, name))
        state_dict_renamed = OrderedDict()
        for k, v in state_dict.items():
            # rename keys
            kn = "module."+k[18:]
            state_dict_renamed[kn] = state_dict[k]

        self.model.load_state_dict(state_dict_renamed)
        # evaluate on random angles  n-times
        print("random_angles")
        collected_stats = None
        for i in range(random_runs):
            stats = self.val_loop()
            collected_stats = self.add_dicts(collected_stats, stats)
            self.log({name+"_randomAng_"+key: val for key, val in stats.items()})
        self.log({"random_average_"+ key: np.average(val) for key, val in collected_stats.items()})

        if last_entry and rotate:
            collected_stats = None
            for i in np.arange(0, 360, degree_interval):
                self.rotate_transform.set_angle_config(i)
                stats = self.val_loop()
                collected_stats = self.add_dicts(collected_stats, stats)
                stats["angle"] = int(i)
                self.log({name + "_rotatingAng_" + key: val for key, val in stats.items()})
            self.log({"rotating_average_" + key: np.average(val) for key, val in collected_stats.items()})

        return None


    @classmethod
    @param('multi_validate.models_folder')
    @param('multi_validate.add_nonrotate_run')
    @param('logging.log_folder')
    @param('logging.wandb_project')
    @param('logging.wandb_run')
    @param('multi_validate.final_chkp_only')
    def evaluate_folder(cls, models_folder, add_nonrotate_run, log_folder, wandb_project, wandb_run, final_chkp_only):
        evaluator = cls(gpu=0)
        evaluator.wandb_api = wandb.Api()
        entity, project = "tuggeluk", wandb_project
        runs = evaluator.wandb_api.runs(entity + "/" + project)
        previous_runs = [run.name for run in runs]
        # iterate trough config paths
        for config in os.listdir(models_folder):
            if "arch:" in config:
                model_arch = config.split("arch:")[-1].split("__")[0]
                evaluator.model, evaluator.scaler = evaluator.create_model_and_scaler(arch=model_arch)

            if add_nonrotate_run:
                rotate_runs = ["nonRotate", "rotate"]
            else:
                rotate_runs = [""]

            for rotate_run in rotate_runs:
                rotate = True
                if rotate_run == "nonRotate":
                    evaluator.rotate_transform.set_angle_config(angle_config=0)
                    rotate = False
                elif rotate_run == "rotate":
                    evaluator.rotate_transform.set_angle_config(angle_config=-1)

                run_name = wandb_run+"_"+config+"_"+rotate_run
                if run_name in previous_runs:
                    print("skipping: "+run_name)
                else:
                    #(Re-)initialize logger for new config
                    evaluator.initialize_logger(os.path.join(log_folder, config),
                                                wandb_project, run_name)

                    config_path = os.path.join(models_folder, config)
                    checkpoints = evaluator.collect_checkpoints(config_path)
                    sorted_keys = list(checkpoints.keys())
                    list.sort(sorted_keys)

                    if final_chkp_only == 1:
                        sorted_keys = [sorted_keys[-1]]

                    for key in sorted_keys:
                        last_entry = key == sorted_keys[-1]
                        evaluator.evaluate_checkpoint(checkpoints[key], config_path, last_entry, rotate)

            #evaluator.eval_and_log()

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
    MultiModelEvaluator.evaluate_folder()
