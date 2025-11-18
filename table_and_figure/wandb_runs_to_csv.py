import pandas as pd
import wandb

def export_wandb_runs_to_csv(entity, project, output_csv="summary.csv"):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    all_data = []
    for run in runs:
        summary = {k: v for k, v in run.summary._json_dict.items() if not k.startswith("_")}
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        row = {"name": run.name, **config, **summary}
        all_data.append(row)

    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv} with {len(df)} rows.")

# Example usage
#export_wandb_runs_to_csv("jackhu0119", "april2_cifar100", "summary_cifar100.csv")
export_wandb_runs_to_csv("jackhu0119", "cifar10_nov", "summary_cifar10.csv")