import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_train_history(run):
    hist = run.history()
    top_1 = np.array(hist["random_average_top_1"])
    top_1 = top_1[~np.isnan(top_1)]
    top_5 = np.array(hist["random_average_top_5"])
    top_5 = top_5[~np.isnan(top_5)]
    return top_1, top_5

def get_config(run):
    load_from_rotate, arch, rotate = run.name.split("__")

    load_from_rotate = load_from_rotate.split(":")[1]
    arch = arch.split(":")[1]
    rotate = rotate.strip("_")
    return load_from_rotate, arch, rotate

def get_avg_performance(runs_df, load_from_rotate, rotate, top_x):
    selection = runs_df[(runs_df.load_from_rotate == load_from_rotate) & (runs_df.rotate == rotate)][top_x]
    selection = np.stack(selection)
    selection = np.average(selection, 0)
    return selection

def generate_avg_plot(runs_df, top_x):
    load_froms = ['0', '1']
    rotates = ['nonRotate', 'rotate']
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    x = np.arange(5, 101, 5)
    for load in load_froms:
        for rotate in rotates:
            avg = get_avg_performance(runs_df, load, rotate, top_x)
            test_rot = str(int(rotate == 'rotate'))
            ax1.plot(x, avg, label="train-rotation:"+load+" test-rotation:"+test_rot)
    ax1.legend()
    ax1.set_ylabel(top_x + ' accuracy')
    ax1.set_xlabel('epoch')
    plt.show()





def main():
    #load data from wandb
    print("")
    wandb_api = wandb.Api()
    entity, project = "tuggeluk", "evaluate_final_base_models"
    runs = wandb_api.runs(entity + "/" + project)

    construct_df = dict()
    for run in runs:
        top_1, top_5 = get_train_history(run)
        load_from_rotate, arch, rotate,  = get_config(run)
        construct_df[run.name] = [load_from_rotate, arch, rotate, top_1, top_5]


    runs_df = pd.DataFrame(construct_df)
    runs_df.index = ["load_from_rotate", "arch", "rotate", "top_1", "top_5"]
    runs_df = runs_df.transpose()

    generate_avg_plot(runs_df, 'top_1')

    ind_models = ["resnet18", "resnet50"]
    print("generate individual plot")







if __name__ == '__main__':
    main()