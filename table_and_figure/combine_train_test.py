import pandas as pd
from table_validation import process_and_compare_summary

base = "table_and_figure/csv/"

def train_results(dataset: str, suffix: str = None):
    """
    Merge training summary and test results with optional suffix.
    """
    # Logic to handle the underscore
    if suffix:
        file_part = f"_{suffix}"
    else:
        file_part = ""

    # Construct filenames dynamically
    train_file = f"wandb_{dataset}{file_part}.csv"
    test_file = f"test_{dataset}{file_part}.csv"
    combined_file = f"{dataset}{file_part}.csv"

    # Load data
    print(f"Processing: {train_file} + {test_file}")
    df1 = pd.read_csv(base + train_file)
    df2 = pd.read_csv(base + test_file)

    # Extract model base name
    df1["name"] = df1["name"].str.split("-").str[0]

    # Count rows where reg == 0 ---
    count_df1 = (df1["reg"] == 0).sum()
    count_df2 = (df2["reg"] == 0).sum()
    print(f"Rows with reg=0 in {train_file}: {count_df1}")
    print(f"Rows with reg=0 in {test_file}: {count_df2}")

    # Set lamb to 0 where reg is 0 (Columns assumed to always exist)
    df1.loc[df1["reg"] == 0, "lamb"] = 0
    df2.loc[df2["reg"] == 0, "lamb"] = 0

    # Merge based on shared keys
    merged_df = pd.merge(df2, df1, on=["name", "reg", "lamb", "celoss"])

    # Save and return
    merged_df.to_csv(base + combined_file, index=False)
    print(f"Merged results saved to: {combined_file}")
    print(f"below is the result for {dataset}:with celoss=1")
    name= f"{dataset}{file_part}" # for name identifier 
    process_and_compare_summary(base + combined_file, name, 1) # Using base + combined_file for the path


# --- Usage Examples ---
# Case 1: With Suffix (Files: wandb_cifar100_lamb_wd.csv)
train_results("cifar10", suffix="lamb_wd")

# Case 2: No Suffix (Files: wandb_cifar100.csv)
# train_results("cifar100")




# for cifar100 data
#print("below is the result for cifar100:")
#process_and_compare_summary("train_results_cifar100.csv", "cifar100_train")