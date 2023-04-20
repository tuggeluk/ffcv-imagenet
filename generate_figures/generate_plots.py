import wandb
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def get_train_history(run,top_1_name, top_5_name):
    hist = run.history()
    top_1 = np.array(hist[top_1_name])
    top_1 = top_1[~np.isnan(top_1)]
    top_5 = np.array(hist[top_5_name])
    top_5 = top_5[~np.isnan(top_5)]
    return top_1, top_5

def get_config(run):
    load_from_rotate = ""
    arch = ""

    if '__' not in run.name:
        arch = run.name
    else:
        for token in run.name.split("__"):
            split = token.split(":")
            if len(split) == 2:
                name, val = split
                if 'random_rotate' in name:
                    load_from_rotate = val

                elif 'arch' in name:
                    arch = val
            elif len(split) > 2:
                arch = split[-1]

    return load_from_rotate, arch

def get_avg_performance(runs_df, load_from_rotate, top_x):
    selection = runs_df[(runs_df.load_from_rotate == load_from_rotate)][top_x]
    selection = np.stack(selection)
    selection = np.average(selection, 0)
    selection = np.append(selection, selection[0])
    return selection

def generate_avg_plot(runs_df, top_x, name='noName.svg', title='noTitle', store_dir=None, show=False):
    load_froms = runs_df.load_from_rotate.unique()
    fig, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
    x = np.arange(0, 361, 360/len(runs_df[top_x].iloc[0]))/180*np.pi
    max_val = 0
    #load_froms = ['0','1']
    for load in np.sort(load_froms):
            avg = get_avg_performance(runs_df, load, top_x)
            if load == '2':
                label = "train-rotation:0 + mental-rot"
                ls = ':'
                co = 'tab:blue'
            else:
                label = "train-rotation:"+load
                if load == '0':
                    ls = '-'
                    co = 'tab:orange'
                else:
                    ls = '-.'
                    co = 'tab:green'

            ax1.plot(x, avg, label=label, ls=ls, color=co, linewidth = 3)
            if max(avg) > max_val:
                max_val = max(avg)
    ax1.set_title(title, fontsize=22)
    # ax1.set_ylabel(top_x + ' accuracy')
    # ax1.set_xlabel('degree')
    #ax1.set_theta_direction(-1)
    #ax1.set_theta_offset(np.pi / 2.0)
    ax1.set_rlabel_position(170)
    ax1.grid(True)

    # for label in ax1.yaxis.get_ticklabels()[1::2]:
    #     label.set_visible(False)
    #plt.yticks(np.arange(0, max_val + 0.5, 0.25))
    ax1.yaxis.set_tick_params(labelsize=14)
    ax1.xaxis.set_tick_params(labelsize=15, pad=10)
    pos = ax1.get_position()
    ax1.set_position([pos.x0, pos.y0, pos.width, pos.height])
    #ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)

    if not store_dir is None:
        plt.savefig(os.path.join(store_dir, name), bbox_inches='tight')
    if show:
        plt.show()

def generate_ind_plot(runs_df, top_x, name='noName.svg', title='noTitle', store_dir=None, show=False):
    load_froms = runs_df.load_from_rotate.unique()
    fig, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
    x = np.arange(0, 361, 360/len(runs_df[top_x].iloc[0]))/180*np.pi
    max_val = 0



    selection = runs_df[(runs_df.load_from_rotate == '2')][top_x]
    train_len = [['50ep_' in x for x in selection.index],['300ep_' in x for x in selection.index]]

    # for i, tr_l in enumerate(train_len):
    #     sel = selection[tr_l]
    #     sel = np.stack(sel)
    #     sel = np.average(sel, 0)
    for i in range(len(selection)):
        sel = selection[i]
        sel = np.append(sel, sel[0])

        name_sel = selection.index[i]

        if '50ep_' in name_sel:
            ls = ':'
        else:
            ls = '-'

        if 'next50' in name_sel or 'net50' in name_sel:
            co = 'tab:orange'
        elif 'next101' in name_sel or 'net152' in name_sel:
            co = 'tab:blue'
        else:
            co = 'tab:green'
        ax1.plot(x, sel, ls=ls, color=co, linewidth=3)

        if max(sel) > max_val:
            max_val = max(sel)

    from matplotlib.lines import Line2D

    if 'resnext' in name_sel:
        custom_lines = [Line2D([0], [0], color='k', lw=4, label = 'AMR 300'),
                        Line2D([2], [2], color='k',  ls=':', lw=4, label='AMR 50'),
                        Line2D([],[],linestyle=''),
                        Line2D([], [], marker="s", markersize=10, linewidth=0, color='tab:orange',  label='ResNext 50'),
                        Line2D([], [], marker="s", markersize=10, linewidth=0, color='tab:blue',  label='ResNext 101'),
                        ]
    else:
        custom_lines = [Line2D([0], [0], color='k', lw=4, label = 'AMR 300'),
                        Line2D([2], [2], color='k',  ls=':', lw=4, label='AMR 50'),
                        Line2D([],[],linestyle=''),
                        Line2D([], [], marker="s", markersize=10, linewidth=0, color='tab:green', label='ResNet 18'),
                        Line2D([], [], marker="s", markersize=10, linewidth=0, color='tab:orange',  label='ResNet 50'),
                        Line2D([], [], marker="s", markersize=10, linewidth=0, color='tab:blue',  label='ResNet 152'),
                        ]

    ax1.legend(handles=custom_lines, bbox_to_anchor=(1.1, 1.05), fontsize=15)
    ax1.set_title(title, fontsize=22)
    # ax1.set_ylabel(top_x + ' accuracy')
    # ax1.set_xlabel('degree')
    #ax1.set_theta_direction(-1)
    #ax1.set_theta_offset(np.pi / 2.0)
    ax1.set_rlabel_position(170)
    ax1.grid(True)

    # for label in ax1.yaxis.get_ticklabels()[1::2]:
    #     label.set_visible(False)
    #plt.yticks(np.arange(0, max_val + 0.5, 0.25))
    ax1.yaxis.set_tick_params(labelsize=14)
    ax1.xaxis.set_tick_params(labelsize=15, pad=10)
    pos = ax1.get_position()
    ax1.set_position([pos.x0, pos.y0, pos.width, pos.height])
    #ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)

    if not store_dir is None:
        plt.savefig(os.path.join(store_dir, name), bbox_inches='tight')
    if show:
        plt.show()

def generate_polar_plot(urls, store_dir, name, wandb_api, restrict_arch=[], title_prefix ="", avg_plot = True):

    runs = wandb_api.runs(urls['base'][0] + "/" + urls['base'][1])
    construct_df = dict()
    for run in runs:
        if 'nonRotate' in run.name:
            continue

        load_from_rotate, arch = get_config(run)
        if len(restrict_arch) > 0 and not arch in restrict_arch:
            continue

        # only use good ViT runs
        if 'vit_b_16' in arch and '_no_mixup_cutmix_' in run.name:
            continue


        top_1, top_5 = get_train_history(run, "epoch_" + str(urls['max_ep']) + "_weights.pt_rotatingAng_top_1",
                                         "epoch_" + str(urls['max_ep']) + "_weights.pt_rotatingAng_top_5")

        construct_df[run.name] = [load_from_rotate, arch, top_1, top_5]


    runs = wandb_api.runs(urls['angleclass'][0] + "/" + urls['angleclass'][1])
    for run in runs:

        load_from_rotate, arch = get_config(run)
        print(arch)
        print(restrict_arch)
        if len(restrict_arch) > 0 and not arch in restrict_arch:
            continue

        top_1, top_5 = get_train_history(run, "top_1_class_corr_pred",
                                         "top_5_class_corr_pred")
        construct_df[run.name] = ['2', arch, top_1, top_5]

    runs_df = pd.DataFrame(construct_df)
    runs_df.index = ["load_from_rotate", "arch", "top_1", "top_5"]
    runs_df = runs_df.transpose()

    # cut out all cnns
    if len(title_prefix) > 0:
        base_name = urls["plot_title"][:-11]
    else:
        base_name = urls["plot_title"]

    if avg_plot:
        generate_avg_plot(runs_df, 'top_1', name=name, title=base_name + title_prefix, store_dir=store_dir, show=True)
    else:
        generate_ind_plot(runs_df, 'top_1', name=name, title=base_name + title_prefix, store_dir=store_dir, show=True)


def generate_plots():

    store_base_dir = 'generate_figures/plots_out'

    wandb_api = wandb.Api()

    urls = dict()

    # urls['ImageNet'] = dict()
    # urls['ImageNet']['base'] = ("tuggeluk", "evaluate_final_base_models_highres")
    # urls['ImageNet']['max_ep'] = 100
    # urls['ImageNet']['angleclass'] = ("tuggeluk", "evaluate_final_angle_class_highres")
    # urls['ImageNet']['training'] = ("tuggeluk", "evaluate_final_base_models")
    # urls['ImageNet']['plot_title'] = ("ImageNet - all CNNs")
    #
    # urls['ImageNetViT'] = dict()
    # urls['ImageNetViT']['base'] = ("tuggeluk", "evaluate_final_base_models_highres_ViT")
    # urls['ImageNetViT']['max_ep'] = 299
    # urls['ImageNetViT']['angleclass'] = ("tuggeluk", "evaluate_angle_class_ViT")
    # urls['ImageNetViT']['training'] = None
    # urls['ImageNetViT']['plot_title'] = ("ImageNet - ViT-16")


    urls['StanfordCars'] = dict()
    urls['StanfordCars']['base'] = ("tuggeluk", "evaluate_final_base_models_highres_StanfordCars")
    urls['StanfordCars']['max_ep'] = 1000
    urls['StanfordCars']['angleclass'] = ("tuggeluk", "test_angleclass_stanfordcars")
    urls['StanfordCars']['training'] = None
    urls['StanfordCars']['plot_title'] = ("Stanford Cars - all CNNs")


    # urls['OxfordPet'] = dict()
    # urls['OxfordPet']['base'] = ("tuggeluk", "evaluate_final_base_models_highres_OxfordPet")
    # urls['OxfordPet']['max_ep'] = 3000
    # urls['OxfordPet']['angleclass'] = ("tuggeluk", "test_angleclass_oxfordpets")
    # urls['OxfordPet']['training'] = None
    # urls['OxfordPet']['plot_title'] = ("Oxford Pet - all CNNs")

    if not os.path.exists(store_base_dir):
        os.makedirs(store_base_dir)

    for dataset, urls in urls.items():

        # name = dataset + 'Polar.pdf'
        # generate_polar_plot(urls, store_base_dir, name, wandb_api)

        if dataset in ['ImageNet', 'StanfordCars', 'OxfordPet']:
            # name = 'EfficientNets' + dataset + 'Polar.pdf'
            # generate_polar_plot(urls, store_base_dir, name, wandb_api,
            #                     restrict_arch=['efficientnet_b4', 'efficientnet_b2', 'efficientnet_b0'],
            #                     title_prefix=" - EfficientNets ")
            #
            # name = 'ResNets' + dataset + 'Polar.pdf'
            # generate_polar_plot(urls, store_base_dir, name, wandb_api,
            #                     restrict_arch=['resnet18', 'resnet50', 'resnet152'],
            #                     title_prefix=" - ResNets ")
            #
            # name = 'ResNeXts' + dataset + 'Polar.pdf'
            # generate_polar_plot(urls, store_base_dir, name, wandb_api,
            #                     restrict_arch=['resnext50_32x4d', 'resnext101_32x8d'],
            #                     title_prefix=" - ResNeXts ")

            name = 'ResNeXts-amr' + dataset + 'Polar.pdf'
            generate_polar_plot(urls, store_base_dir, name, wandb_api,
                                restrict_arch=['resnext50_32x4d', 'resnext101_32x8d'],
                                title_prefix=" - ResNeXts AMR", avg_plot=False)

            name = 'ResNets-amr' + dataset + 'Polar.pdf'
            generate_polar_plot(urls, store_base_dir, name, wandb_api,
                                restrict_arch=['resnet18', 'resnet50', 'resnet152'],
                                title_prefix=" - ResNets AMR", avg_plot=False)

if __name__ == '__main__':
    generate_plots()

    # from torchvision.models import resnet18, resnet34, resnet50
    # resnet18 = resnet18()
    # resnet34 = resnet34()
    # resnet34 = resnet34()