from codebase.engine import evaluate_one_epoch
import torch
from torch.utils.data import random_split, DataLoader
from pathlib import Path
from pyhocon import ConfigFactory, ConfigTree
from codebase.main import prepare_for_training
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description="record models metrics for a given dataset")
parser.add_argument("dataset", type=str, help="'cifar10' or 'cifar100'")
args = parser.parse_args()

dataset = args.dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf_path = f'conf/{dataset}.conf'
conf = ConfigFactory.parse_file(conf_path)
#conf100 = ConfigFactory.parse_file('conf/cifar100.conf')

(_,train_loader,val_loader,criterion,optimizer, scheduler,metric_store,states,) = prepare_for_training(conf, 0)


# Step 1: Extract dataset from the original val_loader
full_val_dataset = val_loader.dataset
# Step 2: Define sizes
val_size = int(0.1 * len(full_val_dataset))

test_size = len(full_val_dataset) - val_size
# Step 3: Create a generator with a fixed seed
generator = torch.Generator().manual_seed(42)
# Step 4: Perform deterministic random split
val_subset, test_subset = random_split(full_val_dataset, [val_size, test_size], generator=generator)
batch_size=1000
# Step 5: Create new DataLoaders
loader_val = DataLoader(
    val_subset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True
)

loader_test = DataLoader(
    test_subset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True
)


df = pd.DataFrame()
root = Path(f"output_{dataset}") # root directory

for model_dir in root.iterdir():
    model_name = model_dir.name
    if model_dir.is_dir():
        for config_dir in model_dir.iterdir():
            if config_dir.is_dir():
                # Parse config from folder name
                s = config_dir.name
                if s =="baseline":
                    reg = 0
                    lamb= 0.1
                else:
                    parts = dict(part.split(":") for part in s.split(","))
                    reg = int(parts["reg"])
                    lamb = float(parts["lamb"])
                config={'model':model_name,'reg':reg,'lamb':lamb}
                model_path = config_dir /'epoch_200.pt'
                model = torch.hub.load("chenyaofo/pytorch-cifar-models", f"{dataset}_{model_name}", pretrained=False)
                model = model.to(device)
                model.load_state_dict(torch.load(model_path, weights_only=True))
                model.eval()
                metric_val =evaluate_one_epoch(
                        loader=loader_val,
                            model=model,
                            reg=None,
                             lamb=0.1,
                             epoch=-100,
                              criterion=criterion,
                              optimizer=optimizer,
                              scheduler=scheduler,
                                 use_amp=False,
                                 accmulated_steps=conf.get_int("accmulated_steps"),
                                     device="cuda",
                                     memory_format=getattr(torch, conf.get("memory_format")),
                                    log_interval=conf.get_int("log_interval"),
                                     max_norm=conf.get_float("max_norm"),
                                     output_dir=None)
                metric_test =evaluate_one_epoch(
                                    loader=loader_test,
                                    model=model,
                                    reg=None,
                                    lamb=0.1,
                                    epoch=-100,
                                    criterion=criterion,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    use_amp=False,
                                    accmulated_steps=conf.get_int("accmulated_steps"),
                                    device="cuda",
                                    memory_format=getattr(torch, conf.get("memory_format")),
                                    log_interval=conf.get_int("log_interval"),
                                    max_norm=conf.get_float("max_norm"),
                                    output_dir=None)
                # Rename all keys for test data
                renamed_metric_test = {k.replace("eval/", "test/"): v for k, v in metric_test.items()}
                # combine the dict
                combined = {**config,**metric_val, **renamed_metric_test}
                 # Append to DataFrame
                df = pd.concat([df, pd.DataFrame([combined])], ignore_index=True)

df.to_csv(f"results_{args.dataset}.csv")




