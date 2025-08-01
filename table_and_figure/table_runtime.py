import pandas as pd
import wandb
from tabulate import tabulate

def export_wandb_runs_to_csv(entity, project, output_csv="summary.csv"):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    all_data = []
    for run in runs:
        summary = {k: v for k, v in run.summary._json_dict.items()}
        config = {k: v for k, v in run.config.items() }
        row = {"name": run.name, **config, **summary}
        all_data.append(row)

    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv} with {len(df)} rows.")

#generate the csv file which contrain col for _runtime
#export_wandb_runs_to_csv("jackhu0119", "april1_cifar10", "runtime_cifar10.csv")
#export_wandb_runs_to_csv("jackhu0119", "april2_cifar100", "runtime_cifar100.csv")



df = pd.read_csv("runtime_cifar10.csv")
# add 1 col represent the model type (without the reg and lamb)
df["group"] = df["name"].str.split("-").str[0]
#print(df["group"].unique())
df = df[["name", "group","reg","lamb","_runtime"]] # only keep the related cols to runtime
df.rename(columns={"_runtime": "runtime"}, inplace=True)

df_lamb1 = df[(df["lamb"] == 1) | (df["reg"] == 0)] # keep lamb=1 and baseline
df_lamb1 = df_lamb1[df_lamb1["reg"]!= 8]
df_lamb1.drop(columns=["lamb", "name"], inplace=True)
df_sorted = df_lamb1.sort_values(by=["group","reg"], ascending= [True,True]) # sort rows


#print(tabulate(df_sorted, headers="keys"))

df_reg0 =  df_sorted[df_sorted['reg']==0].drop('reg',axis=1).rename(columns={"runtime": "runtime_reg0"})
df_reg3 =  df_sorted[df_sorted['reg']==3].drop('reg',axis=1).rename(columns={"runtime": "runtime_reg3"})
df_reg4 =  df_sorted[df_sorted['reg']==4].drop('reg',axis=1).rename(columns={"runtime": "runtime_reg4"})
df_reg5 =  df_sorted[df_sorted['reg']==5].drop('reg',axis=1).rename(columns={"runtime": "runtime_reg5"})

merged_df = pd.merge(df_reg0, df_reg3, on='group')
merged_df = pd.merge(merged_df, df_reg4, on='group')
merged_df = pd.merge(merged_df, df_reg5, on='group')
merged_df = merged_df.rename(columns={"group":"model"})
print(tabulate(merged_df, headers="keys"))
latex_code = merged_df.to_latex(index=False, float_format="%.2f", escape=True)  # index=False omits the index column
print(latex_code)
