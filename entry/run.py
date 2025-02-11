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
        'cifar10_resnet20','cifar10_resnet32','cifar10_resnet44','cifar10_resnet56', 
        'cifar10_repvgg_a0','cifar10_repvgg_a1','cifar10_repvgg_a2', 
        'cifar10_mobilenetv2_x0_5','cifar10_mobilenetv2_x0_75','cifar10_mobilenetv2_x1_0','cifar10_mobilenetv2_x1_4',
        ]},
        },
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-sweep")
    wrapped_main = lambda: main(get_args())
         # Start sweep job.
    wandb.agent(sweep_id, function=wrapped_main, count=100)