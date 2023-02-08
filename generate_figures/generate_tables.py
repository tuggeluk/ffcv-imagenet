import wandb


def generate_table(dataset, urls, wandb_ai):

    return None


def generate_tables():

    wandb_api = wandb.Api()

    urls = dict()

    urls['ImageNet'] = dict()
    urls['ImageNet']['base'] = ("tuggeluk", "evaluate_final_base_models_highres")
    urls['ImageNet']['max_ep'] = 100
    urls['ImageNet']['angleclass'] = [("tuggeluk", "evaluate_final_angle_class_highres"), ("tuggeluk", "evaluate_final_angle_class_highres_5ep") ]
    urls['ImageNet']['training'] = ("tuggeluk", "evaluate_final_base_models")

    urls['ImageNetViT'] = dict()
    urls['ImageNetViT']['base'] = ("tuggeluk", "evaluate_final_base_models_highres_ViT")
    urls['ImageNetViT']['max_ep'] = 299
    urls['ImageNetViT']['angleclass'] = ("tuggeluk", "evaluate_angle_class_ViT")


    urls['StanfordCars'] = dict()
    urls['StanfordCars']['base'] = ("tuggeluk", "evaluate_final_base_models_highres_StanfordCars")
    urls['StanfordCars']['max_ep'] = 1000
    urls['StanfordCars']['angleclass'] = ("tuggeluk", "test_angleclass_stanfordcars")

    urls['OxfordPet'] = dict()
    urls['OxfordPet']['base'] = ("tuggeluk", "evaluate_final_base_models_highres_OxfordPet")
    urls['OxfordPet']['max_ep'] = 3000
    urls['OxfordPet']['angleclass'] = ("tuggeluk", "test_angleclass_oxfordpets")

    for dataset, urls in urls.items():
        generate_table(dataset, urls, wandb_ai)


    return None














if __name__ == '__main__':
    generate_tables()