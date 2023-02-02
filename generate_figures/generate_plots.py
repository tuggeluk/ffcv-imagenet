import wandb
import os
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

def generate_avg_plot(runs_df, top_x, name='noNamed', store_dir=None, show=False):
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
    ax1.set_title(name)
    # ax1.set_ylabel(top_x + ' accuracy')
    # ax1.set_xlabel('degree')
    ax1.set_theta_direction(-1)
    ax1.set_theta_offset(np.pi / 2.0)
    ax1.set_rlabel_position(85)
    ax1.grid(True)

    pos = ax1.get_position()
    ax1.set_position([pos.x0, pos.y0, pos.width, pos.height])
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)

    if not store_dir is None:
        plt.savefig(store_dir+'/polar.png')
    if show:
        plt.show()


def generate_polar_plot(urls, store_dir, wandb_api, restrict_arch=[]):

    runs = wandb_api.runs(urls['base'][0] + "/" + urls['base'][1])
    construct_df = dict()
    for run in runs:
        if 'nonRotate' in run.name:
            continue
        load_from_rotate, arch = get_config(run)
        if len(restrict_arch) > 0 and not arch in restrict_arch:
            continue

        if urls['max_ep'] == 3000:
            print('hold up')

        top_1, top_5 = get_train_history(run, "epoch_" + str(urls['max_ep']) + "_weights.pt_rotatingAng_top_1",
                                         "epoch_" + str(urls['max_ep']) + "_weights.pt_rotatingAng_top_5")

        construct_df[run.name] = [load_from_rotate, arch, top_1, top_5]


    runs = wandb_api.runs(urls['angleclass'][0] + "/" + urls['angleclass'][1])
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

    generate_avg_plot(runs_df, 'top_1', name='hello', store_dir=store_dir, show=True)



def generate_training_plot(urls, store_dir, wandb_api, restrict_arch=[]):
    print("generate polar plots")


def generate_plots():

    store_base_dir = 'generate_figures/plots_out'

    wandb_api = wandb.Api()

    urls = {'ImageNet': dict(), 'ImageNet_ViT': dict(), 'StanfordCars': dict(), 'OxfordPet': dict()}

    urls['ImageNet']['base'] = ("tuggeluk", "evaluate_final_base_models_highres")
    urls['ImageNet']['max_ep'] = 100
    urls['ImageNet']['angleclass'] = ("tuggeluk", "evaluate_final_angle_class_highres")
    urls['ImageNet']['training'] = ("tuggeluk", "evaluate_final_base_models")

    urls['ImageNet_ViT']['base'] = ("tuggeluk", "evaluate_final_base_models_highres_ViT")
    urls['ImageNet_ViT']['max_ep'] = 299
    urls['ImageNet_ViT']['angleclass'] = ("tuggeluk", "evaluate_angle_class_ViT")
    urls['ImageNet_ViT']['training'] = None

    urls['StanfordCars']['base'] = ("tuggeluk", "evaluate_final_base_models_highres_StanfordCars")
    urls['StanfordCars']['max_ep'] = 1000
    urls['StanfordCars']['angleclass'] = ("tuggeluk", "test_angleclass_stanfordcars")
    urls['StanfordCars']['training'] = None

    urls['OxfordPet']['base'] = ("tuggeluk", "evaluate_final_base_models_highres_OxfordPet")
    urls['OxfordPet']['max_ep'] = 3000
    urls['OxfordPet']['angleclass'] = ("tuggeluk", "test_angleclass_oxfordpets")
    urls['OxfordPet']['training'] = None


    for dataset, urls in urls.items():
        store_dir = os.path.join(store_base_dir, dataset)
        if not os.path.exists(store_dir):
            os.makedirs(store_dir)

        generate_polar_plot(urls, store_dir, wandb_api)
        generate_training_plot(urls, store_dir, wandb_api)




if __name__ == '__main__':
    generate_plots()