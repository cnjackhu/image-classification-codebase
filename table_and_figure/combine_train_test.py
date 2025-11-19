import pandas as pd
from table_validation import process_and_compare_summary

def train_results(dataset: str):
    """
    Merge training summary and test results for a given dataset.

    Parameters:
    -----------
    dataset : str
        Dataset name (e.g., "cifar10").
    results_file : str
        Path to the test results CSV (e.g., "results_cifar10.csv").

    Returns:
    --------
    pd.DataFrame
        The merged DataFrame saved to "train_results_<dataset>.csv".
    """
    summary_file = f"summary_{dataset}.csv"
    results_file = f"results_{dataset}.csv"
    output_file = f"train_results_{dataset}.csv"

    # Load data
    df1 = pd.read_csv(summary_file)
    df2 = pd.read_csv(results_file)

    # Extract model base name
    df1["name"] = df1["name"].str.split("-").str[0]

    # Keep only selected columns
    selected_cols = [
        "name", "reg", "lamb",
        "train/top1_acc", "train/top5_acc", "train/L",
        "train/var", "train/ece"
    ]
    df1 = df1[selected_cols]

    # Drop test metrics from results
    drop_cols = [
        "test/top1_acc", "test/top5_acc", "test/L",
        "test/var", "test/ece"
    ]
    df2 = df2.drop(columns=[col for col in drop_cols if col in df2.columns])
    # set value for df1 and df2 where reg=0,lamb=0
    df1.loc[df1["reg"] == 0, "lamb"] = 0
    df2.loc[df2["reg"] == 0, "lamb"] = 0
    # Merge based on shared keys
    merged_df = pd.merge(df2, df1, on=["name", "reg", "lamb"])

    # Rename train/ metrics to test/
    merged_df.rename(
        columns=lambda col: col.replace("train/", "test/") if col.startswith("train/") else col,
        inplace=True
    )

    # Save and return
    merged_df.to_csv(output_file, index=False)
    print(f"Merged results saved to: {output_file}")
    #return merged_df
# Only need to specify dataset 

train_results("cifar10")
train_results("cifar100")

# for cifar10 data
print("below is the result for cifar10:")
process_and_compare_summary("train_results_cifar10.csv", "cifar10_train")

# for cifar100 data
print("below is the result for cifar100:")
process_and_compare_summary("train_results_cifar100.csv", "cifar100_train")