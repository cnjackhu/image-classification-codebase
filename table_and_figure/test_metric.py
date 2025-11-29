import torch
from pathlib import Path
from pyhocon import ConfigFactory, ConfigTree
import pandas as pd
import argparse
import torch.nn as nn
import sys
import os
# Assuming this script is located in a subdirectory (e.g., table_and_figure)
# We add the project root (one level up) to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
# --- END PATH FIX ---
from codebase.main import prepare_for_training
from codebase.engine import evaluate_one_epoch

def run_model_evaluation(dataset_name: str, use_celoss: int, root_dir: str):
    """
    Evaluates models for a given dataset and loss configuration.
    
    NOTE: This version removes all error handling and assumes files and 
    configurations exist.

    Args:
        dataset_name (str): 'cifar10' or 'cifar100'.
        use_celoss (int): 0 or 1. If 1, uses MultiMarginLoss; otherwise uses CrossEntropyLoss (default).
        root_dir (str): The path to the root directory containing model checkpoints.
    """
    print(f"--- Starting Evaluation: Dataset={dataset_name}, CELoss={use_celoss} ---")

    # 1. Setup Configuration (No try/except)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf_path = f'conf/{dataset_name}.conf'
    conf = ConfigFactory.parse_file(conf_path)

    # 2. Prepare Environment (No try/except)
    # We pass '0' as the rank/local_rank if not doing distributed training here
    (_, train_loader, val_loader, criterion, optimizer, scheduler, metric_store, states) = prepare_for_training(conf, 0)
    # 3. Override Loss Function if requested
    if use_celoss == 0:
        print("Overriding criterion: Using MultiMarginLoss")
        criterion = nn.MultiMarginLoss() 
    else:
        print("Using default criterion from prepare_for_training (CrossEntropyLoss)")


    # 4. Setup Paths
    df = pd.DataFrame()
    # Use the input argument directly
    root = Path(root_dir)

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
                    model = torch.hub.load("chenyaofo/pytorch-cifar-models", f"{dataset_name}_{model_name}", pretrained=False)
                    model = model.to(device)
                    model.load_state_dict(torch.load(model_path, weights_only=True))
                    model.eval()
                    metric_test =evaluate_one_epoch(
                                        loader=val_loader,
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
                    combined = {**config,**renamed_metric_test}
                    # Append to DataFrame
                    df = pd.concat([df, pd.DataFrame([combined])], ignore_index=True)
    print(f"--- Finished Evaluation: Dataset={dataset_name}, CELoss={use_celoss} ---")
    return df



if __name__ == "__main__":
    # Example usage via CLI
    #parser = argparse.ArgumentParser(description="Evaluate models for a given dataset")
    #parser.add_argument("dataset", type=str, help="'cifar10' or 'cifar100'")
    #parser.add_argument("celoss", type=int, help="0 for CrossEntropy, 1 for MultiMarginLoss")
    # Added new required argument for the model checkpoint root directory
    #parser.add_argument("root_dir", type=str, help="Root directory containing model checkpoints (e.g., output_cifar10)")
    
    #args = parser.parse_args()
    
    # Pass the new argument to the function
    #run_model_evaluation(args.dataset, args.celoss, args.root_dir)
    #df0 = run_model_evaluation('cifar10',0,'output10')
    df1 = run_model_evaluation('cifar100',1,'output_100')
    #after above combine the 2 csv file
    # Load the two CSV files
    #df0 = pd.read_csv("table_and_figure/csv/test_cifar10+celoss_0.csv", index_col=0)
    #df1 = pd.read_csv("table_and_figure/csv/test_cifar10+celoss_1.csv", index_col=0)

    # Add the indicator column
    #df0["celoss"] = 0
    df1["celoss"] = 1
    df1 = df1.rename(columns={"model": "name"})
    # Concatenate the datasets
    #df_combined = pd.concat([df0, df1], axis=0, ignore_index=True)
    #df_combined = df_combined.rename(columns={"model": "name"})

   
    # Absolute path to the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Path to the 'csv' folder beside the script
    csv_dir = os.path.join(script_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    # Full output path
    output_path = os.path.join(csv_dir,"test_cifar100.csv")
    df1.to_csv(output_path, index=False)