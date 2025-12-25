import wandb
import argparse


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




