import pandas as pd
from tabulate import tabulate
import os


def compare_metrics(df_base: pd.DataFrame, df_best: pd.DataFrame) -> dict:
    # Merge the two DataFrames on 'group'
    merged_df = pd.merge(df_base, df_best, on="group", suffixes=("_base", "_best"))

    # Drop name columns if they exist
    merged_df = merged_df.drop(columns=["name_base", "name_best"], errors="ignore")

    # Initialize a dictionary to store the counts
    counts = {}

    # Accuracy metrics (higher is better)
    counts["top1_acc"] = (merged_df["top1_acc_best"] > merged_df["top1_acc_base"]).sum()
    counts["top5_acc"] = (merged_df["top5_acc_best"] > merged_df["top5_acc_base"]).sum()

    # Error metrics (lower is better)
    counts["L"] = (merged_df["L_best"] < merged_df["L_base"]).sum()
    counts["var"] = (merged_df["var_best"] < merged_df["var_base"]).sum()
    counts["ece"] = (merged_df["ece_best"] < merged_df["ece_base"]).sum()

    # Print results
    for metric, count in counts.items():
        direction = ">" if "acc" in metric else "<"
        print(f"Rows where test/{metric}_best {direction} test/{metric}_base: {count}")
    print("\n" + "-" * 50 + "\n")
    return counts


def df_to_latex_with_bold(df: pd.DataFrame, metric_pairs: list) -> str:
    """
    Convert DataFrame to LaTeX format with boldface for best values in each pair.
    
    Args:
        df: DataFrame to convert
        metric_pairs: List of tuples containing column pairs to compare (base, best)
    
    Returns:
        str: LaTeX formatted table content (without tabular environment)
    """
    # Create a copy of the DataFrame to avoid modifying the original
    latex_df = df.copy()
    
    # For each pair, determine which value is better and make it bold
    for base_col, best_col in metric_pairs:
        if 'acc' in base_col:  # For accuracy metrics, higher is better
            mask = latex_df[best_col] > latex_df[base_col]
        else:  # For error metrics (L, var, ece), lower is better
            mask = latex_df[best_col] < latex_df[base_col]
        
        # Apply bold formatting
        latex_df[base_col] = latex_df[base_col].apply(lambda x: f"{x:.3f}")
        latex_df[best_col] = latex_df[best_col].apply(lambda x: f"{x:.3f}")
        
        # Make the better values bold
        latex_df.loc[mask, best_col] = latex_df.loc[mask, best_col].apply(lambda x: f"\\textbf{{{x}}}")
        latex_df.loc[~mask, base_col] = latex_df.loc[~mask, base_col].apply(lambda x: f"\\textbf{{{x}}}")
    
    # Convert to LaTeX
    latex_table = latex_df.to_latex(
        escape=False,
        float_format=lambda x: f"{x:.3f}",
        index=False,
        column_format='r' * len(df.columns),
        header=False  # Don't include the header
    )
    
    # Remove the tabular environment
    latex_table = latex_table.replace('\\begin{tabular}{' + 'r' * len(df.columns) + '}\n', '')
    latex_table = latex_table.replace('\\end{tabular}', '')
    latex_table = latex_table.replace('\\toprule', '')
    latex_table = latex_table.replace('\\midrule', '')
    latex_table = latex_table.replace('\\bottomrule', '')
    # Replace underscores with escaped underscores in network names
    latex_table = latex_table.replace('_', '\\_')
    
    # Remove any remaining newlines at the start and end
    latex_table = latex_table.strip()
    
    return latex_table


def prepare_filtered_df(df_base: pd.DataFrame, df_best: pd.DataFrame, output_filename: str) -> pd.DataFrame:
    # Merge base and best on 'group'
    merged_df = pd.merge(df_base, df_best, on="group", suffixes=("_base", "_best"))

    # Drop redundant name columns if present
    merged_df = merged_df.drop(columns=["name_base", "name_best"], errors="ignore")
    
    # Reorder columns to group base and best metrics together
    metric_pairs = [
        ("top1_acc_base", "top1_acc_best"),
        ("top5_acc_base", "top5_acc_best"),
        ("L_base", "L_best"),
        ("var_base", "var_best"),
        ("ece_base", "ece_best")
    ]
    
    # Create new column order
    new_columns = ["group"] + [col for pair in metric_pairs for col in pair]
    merged_df = merged_df[new_columns]
    
    print("\nLaTeX big table:")
    latex_table = df_to_latex_with_bold(merged_df, metric_pairs)
    print(latex_table)
    
    # Save big table to file
    os.makedirs("tables", exist_ok=True)
    with open(f"tables/{output_filename}_big.tex", "w") as f:
        f.write(latex_table)

    # Define the valid groups
    valid_groups = [
        "resnet32",
        "vgg19_bn",
        "mobilenetv2_x0_75",
        "shufflenetv2_x0_5",
        "repvgg_a0",
    ]
    
    # Filter for valid groups without setting index
    filtered_df = merged_df[merged_df["group"].isin(valid_groups)]
    
    # Generate and print LaTeX table
    print("\nLaTeX small table:")
    latex_table = df_to_latex_with_bold(filtered_df, metric_pairs)
    print(latex_table)
    
    # Save small table to file
    with open(f"tables/{output_filename}_small.tex", "w") as f:
        f.write(latex_table)
    
    return filtered_df


def create_summary_latex_table(comparison_results: dict, dataset_name: str, lambda_values: list) -> str:
    """
    Create a LaTeX table summarizing the comparison results.
    
    Args:
        comparison_results: Dictionary containing comparison results for different configurations
        dataset_name: Name of the dataset (e.g., "CIFAR-10")
        lambda_values: List of lambda values to include in the table
    
    Returns:
        str: LaTeX formatted table content (without tabular environment)
    """
    # Start with just the content
    latex_table = ""
    
    # Add dataset name and results
    latex_table += f"{dataset_name}     & Best       & Best    &  {comparison_results['all']['top1_acc']}/19     & {comparison_results['all']['top5_acc']}/19     & {comparison_results['all']['L']}/19   & {comparison_results['all']['var']}/19     & {comparison_results['all']['ece']}/19 \\\\\n"
    latex_table += f"    & 0.0     & Best    &  {comparison_results['reg3']['top1_acc']}/19     & {comparison_results['reg3']['top5_acc']}/19     & {comparison_results['reg3']['L']}/19   & {comparison_results['reg3']['var']}/19     & {comparison_results['reg3']['ece']}/19 \\\\\n"
    latex_table += f"    & 0.5    & Best    &  {comparison_results['reg5']['top1_acc']}/19     & {comparison_results['reg5']['top5_acc']}/19     & {comparison_results['reg5']['L']}/19   & {comparison_results['reg5']['var']}/19     & {comparison_results['reg5']['ece']}/19 \\\\\n"
    latex_table += f"    & 1.0    & Best    &  {comparison_results['reg4']['top1_acc']}/19     & {comparison_results['reg4']['top5_acc']}/19     & {comparison_results['reg4']['L']}/19   & {comparison_results['reg4']['var']}/19     & {comparison_results['reg4']['ece']}/19 \\\\\n"
    
    # Add rows for each lambda value
    for lamb in lambda_values:
        latex_table += f"   & Best & {lamb} &  {comparison_results[f'lambda_{lamb}']['top1_acc']}/19     &  {comparison_results[f'lambda_{lamb}']['top5_acc']}/19     & {comparison_results[f'lambda_{lamb}']['L']}/19   & {comparison_results[f'lambda_{lamb}']['var']}/19     & {comparison_results[f'lambda_{lamb}']['ece']}/19 \\\\\n"
    
    return latex_table


def process_and_compare_summary(file_path: str, name: str,celoss:int):
    # Load and preprocess
    df = pd.read_csv(file_path)
    # select if celoss or margin loss
    print(f"celoss is {celoss}")
    df = df[df["celoss"] == celoss]

    df["group"] = df["name"].str.split("-").str[0]

    selected_cols = [
        "name",
        "group",
        "reg",
        "lamb",
        "test/top1_acc",
        "test/top5_acc",
        "test/L",
        "test/var",
        "test/ece",
        "eval/L",
    ]
    df = df[selected_cols]

    # short the rows of df by name column
    df = df.sort_values(by="name")

    df.rename(
        columns={
            "test/top1_acc": "top1_acc",
            "test/top5_acc": "top5_acc",
            "test/L": "L",
            "test/var": "var",
            "test/ece": "ece",
            "eval/L": "L_eval",
        },
        inplace=True,
    )

    # Split into baseline and regularized sets
    base_df = df[df["reg"] == 0].copy()
    base_df.drop(columns=["reg", "lamb"], inplace=True)

    reg_df = df[df["reg"] != 0]
    reg_df = reg_df[reg_df["reg"] != 8]
    #reg_df = reg_df[reg_df["reg"] != 3]
    
    #reg_df = reg_df[reg_df["lamb"] != 0.001]
    #reg_df = reg_df[reg_df["lamb"] != 0.5]
    reg_df = reg_df[reg_df["lamb"] != 2.0]

    reg3_df = reg_df[reg_df["reg"] == 3]
    reg4_df = reg_df[reg_df["reg"] == 4]
    reg5_df = reg_df[reg_df["reg"] == 5]

    # Get best by lowest L in each group
    reg_df_best = reg_df.loc[reg_df.groupby("group")["L_eval"].idxmin()]
    reg3_df_best = reg3_df.loc[reg3_df.groupby("group")["L_eval"].idxmin()]
    reg4_df_best = reg4_df.loc[reg4_df.groupby("group")["L_eval"].idxmin()]
    reg5_df_best = reg5_df.loc[reg5_df.groupby("group")["L_eval"].idxmin()]

    # show filtered df
    print("\n" + "-" * 50 + "\n")
    prepare_filtered_df(base_df, reg_df_best, name)

    # Compare and collect results
    comparison_results = {}
    
    print("\n" + "-" * 50 + "\n")
    print("Best among all regularizer:")
    comparison_results['all'] = compare_metrics(base_df, reg_df_best)

    print("Best among reg=3:")
    comparison_results['reg3'] = compare_metrics(base_df, reg3_df_best)

    print("Best among reg=4:")
    comparison_results['reg4'] = compare_metrics(base_df, reg4_df_best)

    print("Best among reg=5:")
    comparison_results['reg5'] = compare_metrics(base_df, reg5_df_best)

    # Compare fixed lambda values
    print("\n" + "-" * 50 + "\n")
    print("Comparisons for fixed lambda values:")
    
    # Get all unique lambda values and sort them
    lambda_values = sorted(reg_df["lamb"].unique())
    
    for lamb in lambda_values:
        print(f"\nFor lambda = {lamb}:")
        lambda_df = reg_df[reg_df["lamb"] == lamb]
        lambda_df = lambda_df.loc[lambda_df.groupby("group")["L_eval"].idxmin()]
        print(f"Best among lambda = {lamb}")
        comparison_results[f'lambda_{lamb}'] = compare_metrics(base_df, lambda_df)
    
    # Create and save summary table
    summary_table = create_summary_latex_table(comparison_results, name.upper(), lambda_values)
    with open(f"tables/{name}_lambda_summary.tex", "w") as f:
        f.write(summary_table)
    print("\nSummary table content:")
    print(summary_table)

    summary_table = create_summary_latex_table(comparison_results, name.upper(), [])
    with open(f"tables/{name}_summary.tex", "w") as f:
        f.write(summary_table)
    print("\nSummary table content:")
    print(summary_table)


