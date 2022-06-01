#!/bin/bash
# ./run_config_folder.sh rn18_configs/* /home/ubuntu/ImageNet_ffcv 12 /home/ubuntu/rotation_module/ffcv-imagenet/logs
#configs_path="rn18_configs/*"
for FILE in ${1}; do
  python train_imagenet.py \
  --config-file $FILE \
  --data.train_dataset=${2}/train.ffcv \
  --data.val_dataset=${2}/train.ffcv \
  --data.num_workers=${3} \
  --data.in_memory=1 \
  --logging.folder=${4}/$FILE
  done

