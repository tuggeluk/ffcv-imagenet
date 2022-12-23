from collections import OrderedDict
import os
import socket


hostname = socket.gethostname()
on_dgx = 'dgx' in hostname

configs_dict = OrderedDict()


configs_dict["--config-file"] = "configs/angleclass_configs/rn18_angleclass_base.yaml"
configs_dict["--data.train_dataset"] = "/home/ubuntu/OxfordPet_ffcv/train.ffcv"
configs_dict["--data.val_dataset"] = "/home/ubuntu/OxfordPet_ffcv/val.ffcv"
checkpoints_basedir = "/home/ubuntu/ffcv-imagenet/logs/base_models_oxfordpet"
logging_basedir = "/home/ubuntu/rotation_module/ffcv-imagenet/logs/train_angleclass_OxfordPet"
run_name_prefix = ""


configs_dict["--data.num_workers"] = 12
configs_dict["--data.in_memory"] = 1
configs_dict["--data.dataset"] = 'OxfordPet'
configs_dict["--logging.wandb_dryrun"] = 0
configs_dict["--logging.wandb_project"] = "train_angleclass_no_lossshape_OxfordPet"
#configs_dict["--logging.wandb_run"] = ""



#configs_dict["--training.load_from"] = ["_mask_norotate"]
configs_dict["--angleclassifier.freeze_base"] = 1
configs_dict["--lr.lr"] = 0.5
configs_dict["--training.load_from"] = ["arch:efficientnet_b0__random_rotate:0__",
"arch:efficientnet_b2__random_rotate:0__",
"arch:efficientnet_b4__random_rotate:0__",
"arch:resnet18__random_rotate:0__",
"arch:resnet50__random_rotate:0__",
"arch:resnet152__random_rotate:0__",
"arch:resnext50_32x4d__random_rotate:0__",
"arch:resnext101_32x8d__random_rotate:0__"]

configs_dict["--angleclassifier.attach_upright_classifier"] = 1
configs_dict["--angleclassifier.attach_ang_classifier"] = 1
configs_dict["--angleclassifier.classifier_upright"] = ['deep']
configs_dict["--angleclassifier.classifier_ang"] = ['deep']
configs_dict["--angleclassifier.loss_scope"] = 1

configs_dict["--angleclassifier.flatten"] = 'basic'

configs_dict["--angle_testmode.corr_pred"] = [1]
configs_dict["--angleclassifier.shape_class_loss"] = [0]


configs_dict["--data.in_memory"] = 1
configs_dict["--training.epochs"] = [1000, 150]
configs_dict["--training.checkpoint_interval"] = 150

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
                elif k == '--resolution.max_res':
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
