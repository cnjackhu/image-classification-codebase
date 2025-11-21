import os
import pandas as pd
import wandb

def export_wandb_runs_to_csv(entity, project, output_csv="summary.csv"):
    """
    Retrieve Weights & Biases (W&B) runs and export their configurations and
    summary metrics to a CSV file. The CSV file is always saved to a 'csv'
    directory located in the same folder as this script, regardless of the
    current working directory.
    """
    # Absolute path to the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to the 'csv' folder beside the script
    csv_dir = os.path.join(script_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    # Full output path
    output_path = os.path.join(csv_dir, output_csv)

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
# Example usage:
# export_wandb_runs_to_csv("jackhu0119", "april2_cifar100", "summary_cifar100.csv")
export_wandb_runs_to_csv("jackhu0119", "cifar100_nov", "wandb_cifar100.csv")