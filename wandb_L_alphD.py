import wandb
import pandas as pd
#jackhu0119/ablation2/y6v3mpxx
api = wandb.Api()
entity,project = "jackhu0119", "ablation2"
runs = api.runs(f"{entity}/{project}")
metrics=["eval/L","eval/alphaD"]
target_names = ["resnet56-reg:5,lamb:0.1", "resnet56-baseline"]
df_l = pd.DataFrame()
df_ir=pd.DataFrame()
for run in runs:
    name = run.name
    if name in target_names:
        new_name = "SGD"
        if run.config['wd'] != 0:
            new_name += "-L2"
        if run.config['data_aug'] == True:
            new_name += "-DA"
        if run.config['reg'] != 0:
            new_name += "-IR"
        
        history = run.history(keys=metrics)
        df_l[new_name] = history["eval/L"].values
        df_ir[new_name] = history["eval/alphaD"].values
breakpoint()
df_l.to_csv("L.csv", index=False)
df_ir.to_csv("alphaD.csv",index=False)

