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
    train_file = f"wandb_{dataset}.csv"
    #results_file = f"results_{dataset}.csv"
    test_file = f"test_{dataset}.csv"
    combined_file = f"{dataset}.csv"

    # Load data
    df1 = pd.read_csv(base+train_file)
    df2 = pd.read_csv(base+test_file)

    # Extract model base name
    df1["name"] = df1["name"].str.split("-").str[0]



    # set value for df1 and df2 where reg=0,lamb=0
    df1.loc[df1["reg"] == 0, "lamb"] = 0
    df2.loc[df2["reg"] == 0, "lamb"] = 0
    # Merge based on shared keys
    merged_df = pd.merge(df2, df1, on=["name", "reg", "lamb","celoss"])


    # Save and return
    merged_df.to_csv(base+combined_file, index=False)
    print(f"Merged results saved to: {combined_file}")
    #return merged_df
# Only need to specify dataset 
base="table_and_figure/csv/"
train_results("cifar100")
#train_results("cifar100")

# for cifar10 data
print("below is the result for cifar10:with celoss=1")
process_and_compare_summary(base+"cifar100.csv", "cifar100_train",1)

# for cifar100 data
#print("below is the result for cifar100:")
#process_and_compare_summary("train_results_cifar100.csv", "cifar100_train")