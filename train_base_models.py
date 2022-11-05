from collections import OrderedDict
import os
import socket


hostname = socket.gethostname()
on_dgx = 'dgx' in hostname

configs_dict = OrderedDict()

if on_dgx:
    configs_dict["--config-file"] = "configs/base_models/base_models_vit_100_epochs.yaml"
    configs_dict["--data.train_dataset"] = "/cluster/data/tugg/ImageNet_ffcv/train.ffcv"
    configs_dict["--data.val_dataset"] = "/cluster/data/tugg/ImageNet_ffcv/val.ffcv"
    logging_basedir = "/cluster/home/tugg/rotation_module/ffcv-imagenet/logs/base_models_vit"
    run_name_prefix = ""
else:
    configs_dict["--config-file"] = "configs/base_models/base_models_100_epochs.yaml"
    configs_dict["--data.train_dataset"] = "/home/ubuntu/Stanford_ffcv/train.ffcv"
    configs_dict["--data.val_dataset"] = "/home/ubuntu/Stanford_ffcv/val.ffcv"
    logging_basedir = "/home/ubuntu/ffcv-imagenet/logs/base_models_stanfordcars"
    run_name_prefix = ""


configs_dict["--data.num_workers"] = 12
configs_dict["--data.in_memory"] = 1
configs_dict["--data.dataset"] = 'StanfordCars'
configs_dict["--logging.wandb_dryrun"] = 0
configs_dict["--logging.wandb_project"] = "train_base_models_stanfordcars"
configs_dict["--model.arch"] = ['resnet18', 'resnet50', 'resnet152', 'efficientnet_b0', 'efficientnet_b2', 'efficientnet_b4',
                                'resnext50_32x4d', 'resnext101_32x8d']
configs_dict["--training.random_rotate"] = [0, 1]
#configs_dict["--model.arch"] = ['vit_b_32', 'vit_l_32']

#configs_dict["--training.epochs"] = 1

def extend_commands(commands:list, append:str) -> list:
    for i, command in enumerate(commands):
        commands[i] = command + append
    return commands

def build_training_commands() -> list:
    training_commands = [""]
    wandb_run_names = [run_name_prefix]

    for k,v in configs_dict.items():
        if isinstance(v, list):
            init_len = len(training_commands)
            list_ind = 0
            training_commands = len(v)*training_commands
            wandb_run_names = len(v)*wandb_run_names
            for i, vv in enumerate(v):
                if k == '--resolution.max_res':
                    append_cmd = str(k)+"="+str(vv)+" "
                    append_cmd += "--validation.resolution="+str(vv)+" "
                else:
                    append_cmd = str(k)+"="+str(vv)+" "

                append_name = k.split(".")[-1]+":"+str(vv)+"__"
                for _ in range(init_len):
                    training_commands[list_ind] = training_commands[list_ind] + append_cmd
                    wandb_run_names[list_ind] = wandb_run_names[list_ind] + append_name
                    list_ind +=1

        else:
            training_commands = extend_commands(training_commands, str(k)+"="+str(v)+" ")

    for i, command in enumerate(training_commands):
        training_commands[i] = command + "--logging.folder="+os.path.join(logging_basedir, wandb_run_names[i]) + " " +\
        "--logging.wandb_run=" + wandb_run_names[i]

    return training_commands



if __name__ == '__main__':

    training_commands = build_training_commands()
    for command in training_commands:
        final_command = 'python train_imagenet.py '+command
        print(final_command)
        os.system(final_command)
