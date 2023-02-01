import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_train_history(run,top_1_name, top_5_name):
    hist = run.history()
    top_1 = np.array(hist[top_1_name])
    top_1 = top_1[~np.isnan(top_1)]
    top_5 = np.array(hist[top_5_name])
    top_5 = top_5[~np.isnan(top_5)]
    return top_1, top_5

def get_config(run):
    load_from_rotate, arch, _ = run.name.split("__")

    load_from_rotate = load_from_rotate.split(":")[1]
    arch = arch.split(":")[1]

    return load_from_rotate, arch

def get_avg_performance(runs_df, load_from_rotate, top_x):
    selection = runs_df[(runs_df.load_from_rotate == load_from_rotate)][top_x]
    selection = np.stack(selection)
    selection = np.average(selection, 0)
    selection = np.append(selection, selection[0])
    return selection

def generate_avg_plot(runs_df, top_x):
    load_froms = runs_df.load_from_rotate.unique()
    fig, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
    x = np.arange(0, 361, 360/len(runs_df[top_x].iloc[0]))/180*np.pi
    #load_froms = ['0','1']
    for load in load_froms:
            avg = get_avg_performance(runs_df, load, top_x)
            if load == '2':
                label = "train-rotation:0 + angle-classifier"
            else:
                label = "train-rotation:"+load

            ax1.plot(x, avg, label=label)
    ax1.set_title("Top 1 accuracy per rotation angle, averaged across 8 CNN architectures")
    # ax1.set_ylabel(top_x + ' accuracy')
    # ax1.set_xlabel('degree')
    ax1.set_theta_direction(-1)
    ax1.set_theta_offset(np.pi / 2.0)
    ax1.set_rlabel_position(85)
    ax1.grid(True)

    pos = ax1.get_position()
    ax1.set_position([pos.x0, pos.y0, pos.width, pos.height])
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)

    plt.show()





def main():
    wandb_api = wandb.Api()

    # ImageNet - ResNet
    entity, project = "tuggeluk", "evaluate_final_base_models_highres"
    runs = wandb_api.runs(entity + "/" + project)

    restrict_arch = []

    construct_df = dict()
    for run in runs:
        if 'nonRotate' in run.name:
            continue
        load_from_rotate, arch = get_config(run)
        if len(restrict_arch) > 0 and not arch in restrict_arch:
            continue

        top_1, top_5 = get_train_history(run, "epoch_100_weights.pt_rotatingAng_top_1", "epoch_100_weights.pt_rotatingAng_top_5")

        construct_df[run.name] = [load_from_rotate, arch, top_1, top_5]

    add_angleclass_data = True
    if add_angleclass_data:
        entity, project = "tuggeluk", "evaluate_final_angle_class_highres"
        runs = wandb_api.runs(entity + "/" + project)
        for run in runs:
            arch = run.name
            if len(restrict_arch) > 0 and not arch in restrict_arch:
                continue

            top_1, top_5 = get_train_history(run, "top_1_class_corr_pred",
                                             "top_5_class_corr_pred")
            construct_df[run.name] = ['2', arch, top_1, top_5]


    runs_df = pd.DataFrame(construct_df)
    runs_df.index = ["load_from_rotate", "arch", "top_1", "top_5"]
    runs_df = runs_df.transpose()

    generate_avg_plot(runs_df, 'top_1')

    ind_models = ["resnet18", "resnet50"]








if __name__ == '__main__':
    main()