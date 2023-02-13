import wandb
import numpy as np


def read_baseeval_data(urls, wandb_api):
    base_runs = wandb_api.runs(urls[0] + "/" + urls[1])
    construct_df = dict()
    # get top1 class numbers --> first = angl0 performance -> average = roated perofrmance
    # get top1 corr_pred numbers --> average  = angleclass performance
    print('_________________________________________')
    print('_________________________________________')
    print(urls[1])
    print('_________________________________________')
    print('_________________________________________')

    for run in base_runs:
        if '__nonRotate' in run.name:
            continue
        print(run.name)
        hist = run.history()

        top_1 = np.array(hist["epoch_" + str(urls[2]) + "_weights.pt_rotatingAng_top_1"])
        top_1 = top_1[~np.isnan(top_1)]

        print("unrotated performance: "+str(top_1[0]))
        print("rotated performance: "+str(top_1.mean()))
        print('_________________________________________')

    return None


def read_angleclasseval_data(urls, wandb_api):
    base_runs = wandb_api.runs(urls[0] + "/" + urls[1])
    construct_df = dict()
    # get top1 class numbers --> first = angl0 performance -> average = roated perofrmance
    # get top1 corr_pred numbers --> average  = angleclass performance
    print('_________________________________________')
    print(urls[1])
    print('_________________________________________')

    for run in base_runs:

        print(run.config["('model', 'arch')"])
        print(run.name)
        hist = run.history()
        print("unrotated performance: "+str(hist['top_1_class'][0]))
        print("rotated performance: "+str(hist['top_1_class'].mean()))
        print("angleclass performance; "+str(hist['top_1_class_corr_pred'].mean()))

    return None


def generate_tables():

    wandb_api = wandb.Api()

    urls_base = [("tuggeluk", "evaluate_final_base_models", 100), ("tuggeluk", "evaluate_final_base_models_highres_ViT", 299),
                 ("tuggeluk", "evaluate_final_base_models_highres_StanfordCars", 1000), ("tuggeluk", "evaluate_final_base_models_highres_OxfordPet", 3000)]

    for url in urls_base:
        read_baseeval_data(url, wandb_api)


    urls_aclass = [("tuggeluk", "evaluate_final_angle_class_highres"), ("tuggeluk", "evaluate_final_angle_class_highres_5ep"), ("tuggeluk", "evaluate_angle_class_ViT"),
            ("tuggeluk", "test_angleclass_stanfordcars"), ("tuggeluk", "test_angleclass_oxfordpets")]

    for url in urls_aclass:
        read_angleclasseval_data(url, wandb_api)


    return None














if __name__ == '__main__':
    generate_tables()