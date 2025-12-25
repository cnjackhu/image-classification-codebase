#from update_summary_metric import update_wandb_run_summaries
#from wandb_runs_to_csv import export_wandb_runs_to_csv
#from test_metric import run_model_evaluation
#from combine_train_test import combine_train_test_results
import os
import pandas as pd
import wandb
import argparse
import torch
from pathlib import Path
from pyhocon import ConfigFactory
import torch.nn as nn
import sys
from table_validation import process_and_compare_summary
# Assuming this script is located in a subdirectory (e.g., table_and_figure)
# We add the project root (one level up) to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
# --- END PATH FIX ---
from codebase.main import prepare_for_training
from codebase.engine import evaluate_one_epoch

def update_wandb_run_summaries(project: str, entity: str = "jackhu0119", metrics: list[str] = None):
    """
    Update the summary of all runs in a given Weights & Biases project with the last values
    of specified metrics from the history.

    Parameters:
        entity (str): W&B entity (user or team name).
        project (str): W&B project name.
        metrics (list[str], optional): List of metric keys to update. Uses default if None.
    """
    if metrics is None:
        metrics = ["eval/top1_acc", "eval/top5_acc", "eval/L", "eval/loss", "eval/ece", "eval/mce", "eval/var"]
    
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    for run in runs:
        history = run.history(keys=metrics)
        for metric in metrics:
            run.summary[metric] = history[metric].iloc[-1]
        run.summary.update()
        print("Finished update for:", run.name)



def export_wandb_runs_to_csv(project, suffix, entity="jackhu0119"):
    """
    Retrieve Weights & Biases (W&B) runs and export their configurations and
    summary metrics to a CSV file. 
    
    The filename is determined automatically:
    - If project contains 'cifar100' -> wandb_cifar100_{suffix}.csv
    - If project contains 'cifar10'  -> wandb_cifar10_{suffix}.csv
    
    Parameters:
    - project (str): The W&B project name.
    - suffix (str): The suffix for the output filename.
    - entity (str): The W&B entity (user/team). Defaults to "jackhu0119".

    Example usage:
    If project is "cifar100_dec" and suffix is "lamb_wd", 
    output will be "wandb_cifar100_lamb_wd.csv"
    export_wandb_runs_to_csv("cifar10_lamb_wd", "lamb_wd")   
    export_wandb_runs_to_csv("cifar10_sam", "sam")
    export_wandb_runs_to_csv("cifar100_sam", "sam")
    """
    # Absolute path to the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to the 'csv' folder beside the script
    csv_dir = os.path.join(script_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    # Determine output filename based on project name
    # We check cifar100 first because 'cifar100' string contains 'cifar10'
    if "cifar100" in project:
        dataset_name = "cifar100"
    elif "cifar10" in project:
        dataset_name = "cifar10"
    else:
        # Raise an exception if the project name doesn't match expected patterns
        raise ValueError(f"Project name '{project}' must contain 'cifar10' or 'cifar100'.")

    output_csv = f"wandb_{dataset_name}_{suffix}.csv"
    output_path = os.path.join(csv_dir, output_csv)

    print(f"Fetching runs for {project}... Target file: {output_csv}")

    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    all_data = []
    for run in runs:
        summary = {k: v for k, v in run.summary._json_dict.items() if not k.startswith("_")}
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        row = {"name": run.name, **config, **summary}
        all_data.append(row)

    df = pd.DataFrame(all_data)
    df.to_csv(output_path, index=False)

    print(f"Saved to {output_path} with {len(df)} rows.")


def run_model_evaluation(dataset: str, root_dir: str, suffix: str = None, use_celoss: int = 1,):
    """
    Evaluates models, processes results, and saves them to a CSV file.

    Args:
        dataset_name (str): 'cifar10' or 'cifar100'.
        use_celoss (int): 0 or 1. If 0, uses MultiMarginLoss; otherwise uses CrossEntropyLoss.
        root_dir (str): The path to the root directory containing model checkpoints.
        suffix (str, optional): Suffix for the output filename (e.g., 'lamb_wd').
    Example:
        run_model_evaluation(dataset_name='cifar10', root_dir='output_10', suffix='sam')
        run_model_evaluation(dataset_name='cifar100', root_dir='output_100', suffix='sam')
    """
    print(f"--- Starting Evaluation: Dataset={dataset}, CELoss={use_celoss} ---")
    # 1. Setup Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf_path = f'conf/{dataset}.conf'
    conf = ConfigFactory.parse_file(conf_path)

    # 2. Prepare Environment
    (_, train_loader, val_loader, criterion, optimizer, scheduler, metric_store, states) = prepare_for_training(conf, 0)
    
    # 3. Override Loss Function if requested
    if use_celoss == 0:
        print("Overriding criterion: Using MultiMarginLoss")
        criterion = nn.MultiMarginLoss() 
    else:
        print("Using default criterion (CrossEntropyLoss)")

    # 4. Run Evaluation Loop
    df = pd.DataFrame()
    root = Path(root_dir)

    for model_dir in root.iterdir():
        model_name = model_dir.name
        if model_dir.is_dir():
            for config_dir in model_dir.iterdir():
                if config_dir.is_dir():
                    s = config_dir.name
                    if s == "baseline":
                        reg = 0
                        lamb = 0.1
                    else:
                        parts = dict(part.split(":") for part in s.split(","))
                        reg = int(parts["reg"])
                        lamb = float(parts["lamb"])
                    
                    config = {'model': model_name, 'reg': reg, 'lamb': lamb}
                    model_path = config_dir / 'epoch_200.pt'
                    
                    # Load Model
                    model = torch.hub.load("chenyaofo/pytorch-cifar-models", f"{dataset}_{model_name}", pretrained=False)
                    model = model.to(device)
                    model.load_state_dict(torch.load(model_path, weights_only=True))
                    model.eval()
                    
                    # Evaluate
                    metric_test = evaluate_one_epoch(
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
                        output_dir=None
                    )
                    
                    # Rename keys and combine
                    renamed_metric_test = {k.replace("eval/", "test/"): v for k, v in metric_test.items()}
                    combined = {**config, **renamed_metric_test}
                    df = pd.concat([df, pd.DataFrame([combined])], ignore_index=True)

    # 5. Post-Processing (Check Removed)
    df["celoss"] = use_celoss
    df = df.rename(columns={"model": "name"})
    
    print(f"--- Finished Evaluation: Dataset={dataset}, CELoss={use_celoss} ---")

    # 6. Save to CSV
    if suffix:
        filename = f"test_{dataset}_{suffix}.csv"
    else:
        filename = f"test_{dataset}.csv"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(script_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    
    output_path = os.path.join(csv_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


def combine_train_test_results(dataset: str, suffix: str = None):
    """
    Merge training summary and test results with optional suffix.
    """
    # Logic to handle the underscore
    base = "table_and_figure/csv/"
    if suffix:
        suffix = f"_{suffix}"
    else:
        suffix = ""

    # Construct filenames dynamically
    train_file = f"wandb_{dataset}{suffix}.csv"
    test_file = f"test_{dataset}{suffix}.csv"
    combined_file = f"{dataset}{suffix}.csv"

    # Load data
    print(f"Processing: {train_file} + {test_file}")
    df1 = pd.read_csv(base + train_file)
    df2 = pd.read_csv(base + test_file)

    # Extract model base name
    df1["name"] = df1["name"].str.split("-").str[0]

    # Count rows where reg == 0 ---
    count_df1 = (df1["reg"] == 0).sum()
    count_df2 = (df2["reg"] == 0).sum()
    print(f"Rows with reg=0 in {train_file}: {count_df1}")
    print(f"Rows with reg=0 in {test_file}: {count_df2}")

    # Set lamb to 0 where reg is 0 (Columns assumed to always exist)
    df1.loc[df1["reg"] == 0, "lamb"] = 0
    df2.loc[df2["reg"] == 0, "lamb"] = 0

    # Merge based on shared keys
    merged_df = pd.merge(df2, df1, on=["name", "reg", "lamb", "celoss"])

    # Save and return
    output_path = base + combined_file
    if os.path.exists(output_path): #avoid update generated results mistakenly
        print(f"WARNING: File '{output_path}' already exists. Program aborted.")
        sys.exit()  # This stops the script immediately
    
    merged_df.to_csv(output_path, index=False)
    print(f"Merged results saved to: {combined_file}")
    print(f"below is the result for {dataset}:with celoss=1")
    name= f"{dataset}{suffix}" # for name prefix in the generated tex file
    #process_and_compare_summary(output_path, name, 1) # Using base + combined_file for the path




suffix="samm"
dataset="cifar10"
update_wandb_run_summaries(project="cifar10_sam")
export_wandb_runs_to_csv(project="cifar10_sam", suffix=suffix)
run_model_evaluation(dataset=dataset, root_dir='output_10(baseline_sam)', suffix=suffix)
combine_train_test_results(dataset=dataset, suffix = suffix)

