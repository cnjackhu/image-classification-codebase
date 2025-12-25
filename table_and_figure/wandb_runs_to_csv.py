import os
import pandas as pd
import wandb


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

