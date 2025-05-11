import pandas as pd
from tabulate import tabulate




def compare_metrics(df_base: pd.DataFrame, df_best: pd.DataFrame) -> dict:
    # Merge the two DataFrames on 'group'
    merged_df = pd.merge(
        df_base, df_best,
        on="group",
        suffixes=("_base", "_best")
    )

    # Drop name columns if they exist
    merged_df = merged_df.drop(columns=["name_base", "name_best"], errors="ignore")

    # Initialize a dictionary to store the counts
    counts = {}

    # Accuracy metrics (higher is better)
    counts["top1_acc"] = (merged_df["top1_acc_best"] > merged_df["top1_acc_base"]).sum()
    counts["top5_acc"] = (merged_df["top5_acc_best"] > merged_df["top5_acc_base"]).sum()

    # Error metrics (lower is better)
    counts["L"]   = (merged_df["L_best"]   < merged_df["L_base"]).sum()
    counts["var"] = (merged_df["var_best"] < merged_df["var_base"]).sum()
    counts["ece"] = (merged_df["ece_best"] < merged_df["ece_base"]).sum()

    # Print results
    for metric, count in counts.items():
        direction = ">" if "acc" in metric else "<"
        print(f"Rows where eval/{metric}_best {direction} eval/{metric}_base: {count}")

    return counts


# ===================================================================================================

def prepare_filtered_df(df_base: pd.DataFrame, df_best: pd.DataFrame) -> pd.DataFrame:
    # Merge base and best on 'group'
    merged_df = pd.merge(
        df_base, df_best,
        on="group",
        suffixes=("_base", "_best")
    )

    # Drop redundant name columns if present
    merged_df = merged_df.drop(columns=["name_base", "name_best"], errors="ignore")
    print("\n" + "-" * 50 + "\n")
    print("appendix table")
    print(tabulate(merged_df, headers='keys',floatfmt=".3f"))

    # Define the valid groups
    valid_groups = [
        "resnet32",
        "vgg16_bn",
        "mobilenetv2_x0_75",
        "shufflenetv2_x1_0",
        "repvgg_a1"
    ]
    # Reindex and drop columns
    merged_df.reset_index(drop=True, inplace=True)
    merged_df.set_index("group", inplace=True)
    filtered_df =  merged_df.reindex(index=valid_groups).drop(columns=["lamb", "reg"])
    print("\n" + "-" * 50 + "\n")
    print("small table:")
    print(tabulate(filtered_df.head(20), headers='keys',floatfmt=".3f"))
    return filtered_df




def process_and_compare_summary(file_path: str):
    # Load and preprocess
    df = pd.read_csv(file_path)
    df["group"] = df["name"].str.split("-").str[0]
    
    selected_cols = ["name", "group", "reg", "lamb", "eval/top1_acc", "eval/top5_acc", "eval/L", "eval/var", "eval/ece"]
    df = df[selected_cols]
    df.rename(columns={
        "eval/top1_acc": "top1_acc",
        "eval/top5_acc": "top5_acc",
        "eval/L": "L",
        "eval/var": "var",
        "eval/ece": "ece"
    }, inplace=True)

    # Split into baseline and regularized sets
    base_df = df[df["reg"] == 0].copy()
    base_df.drop(columns=["reg", "lamb"], inplace=True)

    reg_df  = df[df["reg"] != 0]
    reg3_df = df[df["reg"] == 3]
    reg4_df = df[df["reg"] == 4]
    reg5_df = df[df["reg"] == 5]

    # Get best by lowest L in each group
    reg_df_best  = reg_df.loc[reg_df.groupby("group")["L"].idxmin()]
    reg3_df_best = reg3_df.loc[reg3_df.groupby("group")["L"].idxmin()]
    reg4_df_best = reg4_df.loc[reg4_df.groupby("group")["L"].idxmin()]
    reg5_df_best = reg5_df.loc[reg5_df.groupby("group")["L"].idxmin()]
    
    #show filtered df
    print("\n" + "-" * 50 + "\n")
    prepare_filtered_df(base_df,reg_df_best)

    # Compare
    print("\n" + "-" * 50 + "\n")
    print("Best among all regularizer:")
    compare_metrics(base_df, reg_df_best)

    print("\n" + "-" * 50 + "\n")
    print("Best among reg=3:")
    compare_metrics(base_df, reg3_df_best)

    print("\n" + "-" * 50 + "\n")
    print("Best among reg=4:")
    compare_metrics(base_df, reg4_df_best)

    print("\n" + "-" * 50 + "\n")
    print("Best among reg=5:")
    compare_metrics(base_df, reg5_df_best)

# for cifar10 data
process_and_compare_summary("summary_cifar10.csv")

# for cifar100 data
#process_and_compare_summary("summary_cifar100.csv")
