from collections import OrderedDict
import os
import socket


hostname = socket.gethostname()
on_dgx = 'dgx' in hostname

configs_dict = OrderedDict()

if on_dgx:
    configs_dict["--config-file"] = "configs/angleclass_configs/rn50_angleclass_base.yaml"
    configs_dict["--data.train_dataset"] = "/cluster/data/tugg/ImageNet_ffcv/train.ffcv"
    configs_dict["--data.val_dataset"] = "/cluster/data/tugg/ImageNet_ffcv/val.ffcv"
    #configs_dict["--logging.folder"] = "/cluster/home/tugg/rotation_module/ffcv-imagenet/logs"
    checkpoints_basedir = "logs/rn50_base_configs"
    logging_basedir = "/cluster/home/tugg/rotation_module/ffcv-imagenet/logs"
    run_name_prefix = "rn50_"
else:
    configs_dict["--config-file"] = "configs/angleclass_configs/rn18_angleclass_base.yaml"
    configs_dict["--data.train_dataset"] = "/home/ubuntu/ImageNet_ffcv/train.ffcv"
    configs_dict["--data.val_dataset"] = "/home/ubuntu/ImageNet_ffcv/val.ffcv"
    #configs_dict["--logging.folder"] = "/home/ubuntu/rotation_module/ffcv-imagenet/logs"
    checkpoints_basedir = "logs/rn18_base_configs"
    logging_basedir = "/home/ubuntu/rotation_module/ffcv-imagenet/logs"
    run_name_prefix = "rn18_"


configs_dict["--data.num_workers"] = 12
configs_dict["--data.in_memory"] = 1
configs_dict["--logging.wandb_dryrun"] = 0
configs_dict["--logging.wandb_project"] = "train_anglclass_corr_pred"
#configs_dict["--logging.wandb_run"] = ""


#configs_dict["--training.load_from"] = ["mask_rotate", "_mask_norotate"]
configs_dict["--training.load_from"] = ["_mask_norotate"]
configs_dict["--angleclassifier.freeze_base"] = 1
configs_dict["--lr.lr"] = 0.5

configs_dict["--angleclassifier.attach_upright_classifier"] = 1
configs_dict["--angleclassifier.attach_ang_classifier"] = 1
configs_dict["--angleclassifier.classifier_upright"] = ['deep']
#configs_dict["--angleclassifier.classifier_ang"] = ['deep']
configs_dict["--angleclassifier.classifier_ang"] = ['deep', 'deepx2', 'deepslant']
configs_dict["--angleclassifier.angle_binsize"] = ['lr', 3]

configs_dict["--angleclassifier.flatten"] = 'basic'

configs_dict["--angle_testmode.corr_pred"] = [1]



configs_dict["--data.in_memory"] = 1
configs_dict["--training.epochs"] = 10

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
