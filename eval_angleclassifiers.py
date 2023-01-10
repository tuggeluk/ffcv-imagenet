from collections import OrderedDict
import os
import socket


hostname = socket.gethostname()
on_dgx = 'dgx' in hostname

configs_dict = OrderedDict()


# python /cluster/home/tugg/rotation_module/ffcv-imagenet/train_imagenet.py --config-file configs/rn18_debug_configs/test_angle_class_base.yaml
# --data.train_dataset=/cluster/data/tugg/ImageNet_ffcv/train.ffcv --data.val_dataset=/cluster/data/tugg/ImageNet_ffcv/val.ffcv --data.num_workers=8
# --data.in_memory=1 --training.load_from=logs/AngleClass_no_losshape/load_from:random_rotate:0__arch:resnet18____classifier_upright:deep__classifier_ang:deep__corr_pred:1__shape_class_loss:0__epochs:33__/final_weights.pt
# --model.arch=resnet18 --eval_configs.degree_interval=2 --logging.wandb_run=resnet18 --dist.port=12356

mode = "stanford50"

if mode == "stanford50":
    configs_dict["--data.train_dataset"] = "/home/ubuntu/Stanford_ffcv/train.ffcv"
    configs_dict["--data.val_dataset"] = "/home/ubuntu/Stanford_ffcv/val.ffcv"
    checkpoints_basedir = "/home/ubuntu/ffcv-imagenet/logs/AngleClass50ep_StanfordCars"
    configs_dict["--dist.port"] = 12253
    configs_dict["--logging.wandb_project"] = "test_angleclass_stanfordcars"
    configs_dict["--data.dataset"] = "StanfordCars"
    run_name_prefix = "50ep_"

elif mode == "stanford300":
    configs_dict["--data.train_dataset"] = "/home/ubuntu/Stanford_ffcv/train.ffcv"
    configs_dict["--data.val_dataset"] = "/home/ubuntu/Stanford_ffcv/val.ffcv"
    checkpoints_basedir = "/home/ubuntu/ffcv-imagenet/logs/AngleClass300ep_StanfordCars"
    configs_dict["--dist.port"] = 12254
    configs_dict["--logging.wandb_project"] = "test_angleclass_stanfordcars"
    configs_dict["--data.dataset"] = "StanfordCars"
    run_name_prefix = "300ep_"

elif mode == "oxford150":
    configs_dict["--data.train_dataset"] = "/home/ubuntu/OxfordPet_ffcv/train.ffcv"
    configs_dict["--data.val_dataset"] = "/home/ubuntu/OxfordPet_ffcv/val.ffcv"
    checkpoints_basedir = "/home/ubuntu/ffcv-imagenet/logs"
    configs_dict["--dist.port"] = 12255
    configs_dict["--logging.wandb_project"] = "test_angleclass_oxfordpets"
    configs_dict["--data.dataset"] = "OxfordPet"
    run_name_prefix = "150ep_"

elif mode == "oxford1000":
    configs_dict["--data.train_dataset"] = "/home/ubuntu/OxfordPet_ffcv/train.ffcv"
    configs_dict["--data.val_dataset"] = "/home/ubuntu/OxfordPet_ffcv/val.ffcv"
    checkpoints_basedir = "/home/ubuntu/ffcv-imagenet/logs/AngleClass1000ep_OxfordPet"
    configs_dict["--dist.port"] = 12256
    configs_dict["--logging.wandb_project"] = "test_angleclass_oxfordpets"
    configs_dict["--data.dataset"] = "OxfordPet"
    run_name_prefix = "1000ep_"

else:
    import sys
    sys.exit(1)


configs_dict["--config-file"] = "configs/rn18_debug_configs/test_angle_class_base.yaml"

logging_basedir = "/home/ubuntu/rotation_module/ffcv-imagenet/logs"


configs_dict["--data.num_workers"] = 8
configs_dict["--data.in_memory"] = 1
configs_dict["--logging.wandb_dryrun"] = 0
configs_dict["--eval_configs.degree_interval"] = 2

configs_dict["--angleclassifier.flatten"] = 'basic'

configs_dict["--training.load_from"] = [
"load_from:arch:efficientnet_b0__random_rotate:0",
# "load_from:arch:efficientnet_b2__random_rotate:0",
# "load_from:arch:efficientnet_b4__random_rotate:0",
# "load_from:arch:resnet152__random_rotate:0",
# "load_from:arch:resnet18__random_rotate:0",
# "load_from:arch:resnet50__random_rotate:0",
# "load_from:arch:resnext101_32x8d__random_rotate:0",
# "load_from:arch:resnext50_32x4d__random_rotate:0",
]


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
                if k == '--training.load_from':
                    candidate_configs = [x for x in os.listdir(checkpoints_basedir) if vv in x]
                    assert len(candidate_configs) == 1
                    p = os.path.join(checkpoints_basedir, candidate_configs[0], 'final_weights.pt')
                    append_cmd = str(k)+"="+p+" "
                    if "arch:" in vv:
                        arch_name = vv.split("arch:")[-1].split("__")[0]
                        append_cmd += "--model.arch=" + arch_name + " "



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
