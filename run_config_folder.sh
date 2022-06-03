#!/bin/bash
# ./run_config_folder.sh 'rn18_base_configs/*' /home/ubuntu/ImageNet_ffcv 12 /home/ubuntu/rotation_module/ffcv-imagenet/logs
# ./run_config_folder.sh 'rn50_base_configs/*' /cluster/data/tugg/ImageNet_ffcv 12 /cluster/home/tugg/rotation_module/ffcv-imagenet/logs

# python train_imagenet.py --config-file rn50_base_configs/rn50_88_epochs_mask_rotate.yaml      --data.train_dataset=/cluster/data/tugg/ImageNet_ffcv/train.ffcv     --data.val_dataset=/cluster/data/tugg/ImageNet_ffcv/val.ffcv     --data.num_workers=12 --data.in_memory=1     --logging.folder=/cluster/home/tugg/rotation_module/ffcv-imagenet/logs
# python train_imagenet.py --config-file rn50_base_configs/rn50_88_epochs_mask_rotate.yaml      --data.train_dataset=/cluster/data/tugg/ImageNet_ffcv/train.ffcv     --data.val_dataset=/cluster/data/tugg/ImageNet_ffcv/val.ffcv     --data.num_workers=20 --data.in_memory=1     --logging.folder=/cluster/home/tugg/rotation_module/ffcv-imagenet/logs/rn50_base_configs/rn50_88_epochs_mask_rotate.yaml



#configs_path="rn18_configs/*"
for FILE in ${1}; do
  python train_imagenet.py \
  --config-file $FILE \
  --data.train_dataset=${2}/train.ffcv \
  --data.val_dataset=${2}/val.ffcv \
  --data.num_workers=${3} \
  --data.in_memory=1 \
  --logging.folder=${4}/$FILE
  done

