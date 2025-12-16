from codebase.config import get_args
from codebase.main import main
import wandb

if __name__ == "__main__":
    #main(get_args())
    # Define sweep config
    sweep_configuration = {
        "method": "grid",
        "name": "sweep",
        "parameters": {
            "sweep_model": {"values": [ ]},
        },
    }
    base_models = [
    'resnet20', 'resnet32', 'resnet44', 'resnet56',
    'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
    'repvgg_a0', 'repvgg_a1', 'repvgg_a2',
    'mobilenetv2_x0_5', 'mobilenetv2_x0_75', 'mobilenetv2_x1_0', 'mobilenetv2_x1_4',
    'shufflenetv2_x0_5', 'shufflenetv2_x1_0', 'shufflenetv2_x1_5', 'shufflenetv2_x2_0',
    ]
    args = get_args()
    # Prefix based on number of classes
    num_classes = args.conf.data.num_classes
    is_vit = args.conf.data.is_vit
    if num_classes==1000 or is_vit:
        prefix=""
    elif num_classes == 100:
        prefix = "cifar100_"
    elif num_classes == 10:
        prefix = "cifar10_"
    else:
        raise ValueError("Unsupported number of classes")
    breakpoint()
    selected_models = [
    'vit_base_patch16_224', #'vit_b32', 'vit_h14', 'vit_l16', 'vit_l32', #'vgg16_bn', 'vgg19_bn',
    # Add more if needed
    ]
    #selected_models=base_models
    sweep_configuration["parameters"]["sweep_model"]["values"] = [prefix + model for model in selected_models]
    print(sweep_configuration)
    project_name = args.conf.sweep_name
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)
    wrapped_main = lambda: main(get_args())
         # Start sweep job.
    wandb.agent(sweep_id, function=wrapped_main, count=100)
