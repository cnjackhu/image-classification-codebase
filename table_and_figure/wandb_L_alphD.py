import pandas as pd
import wandb
import sys
import os

# jackhu0119/ablation2/y6v3mpxx
# generate the L.csv and alphaD.csv for plot figure in the paper
api = wandb.Api()
entity, project = "jackhu0119", "ablation_nov"
runs = api.runs(f"{entity}/{project}")
metrics = ["eval/L", "eval/alphaD"]
target_names = ["resnet56-reg:5,lamb:0.1", "resnet56-baseline"]
df_l = pd.DataFrame()
df_ir = pd.DataFrame()

for run in runs:
    name = run.name
    if name in target_names and run.config["lamb"] == 0.1:
        print(name)
        new_name = "SGD"
        if run.config["wd"] > 0:
            new_name += "-L2"
        if run.config["data_aug"] == True:
            new_name += "-DA"
        if run.config["reg"] == 5:
            new_name += "-IR"
        history = run.history(keys=metrics)
        df_l[new_name] = history["eval/L"].values
        df_ir[new_name] = history["eval/alphaD"].values
# Absolute path to the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Path to the 'csv' folder beside the script
csv_dir = os.path.join(script_dir, "csv")
os.makedirs(csv_dir, exist_ok=True)
# Full output path
df_l.to_csv(os.path.join(csv_dir, "L.csv"), index=False)
df_ir.to_csv(os.path.join(csv_dir, "alphaD.csv"), index=False)