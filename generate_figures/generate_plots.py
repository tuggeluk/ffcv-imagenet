import wandb

def generate_plots():

    wandb_api = wandb.Api()

    urls = {'ImageNet': dict(), 'StanfordCars': dict(), 'OxfordPet': dict()}

    urls['ImageNet']['base'] = ("tuggeluk", "evaluate_final_base_models_highres")
    urls['ImageNet']['angleclass'] = ("tuggeluk", "evaluate_final_angle_class_highres")
    urls['ImageNet']['training'] = ("tuggeluk", "evaluate_final_angle_class_highres")

    urls['StanfordCars']['base'] = None
    urls['StanfordCars']['angleclass'] = None

    urls['OxfordPet']['base'] = None
    urls['OxfordPet']['angleclass'] = None

    for dataset, urls in urls.items():
        print("yep")

if __name__ == '__main__':
    generate_plots()