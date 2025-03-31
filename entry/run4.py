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
            "sweep_model": {"values": [
        #'cifar10_resnet20','cifar10_resnet32','cifar10_resnet44','cifar10_resnet56', 
        # 'cifar10_vgg11_bn','cifar10_vgg13_bn','cifar10_vgg16_bn','cifar10_vgg19_bn',
        # 'cifar10_repvgg_a0','cifar10_repvgg_a1','cifar10_repvgg_a2', 
        'cifar10_mobilenetv2_x0_5','cifar10_mobilenetv2_x0_75','cifar10_mobilenetv2_x1_0','cifar10_mobilenetv2_x1_4',
        # 'cifar10_shufflenetv2_x0_5', 'cifar10_shufflenetv2_x1_0','cifar10_shufflenetv2_x1_5','cifar10_shufflenetv2_x2_0',
         
        ]},
        },
    }

    args = get_args()
    project_name = args.conf.sweep_name
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)
    wrapped_main = lambda: main(args)
         # Start sweep job.
    wandb.agent(sweep_id, function=wrapped_main, count=100)