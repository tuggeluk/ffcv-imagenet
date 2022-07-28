import os
import shutil

source = "/home/tugg/Documents/Datasets/ILSVRC2012_img_val/images"
target = "/home/tugg/Documents/Datasets/val_large"

for folder in os.listdir(target):
    fold_path = os.path.join(target, folder)
    for img in os.listdir(fold_path):
        shutil.copy(os.path.join(source, img), os.path.join(fold_path, img))