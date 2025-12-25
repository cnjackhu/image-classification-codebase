import pandas as pd
import numpy as np

def compare_csv_files(file1_path, file2_path, tolerance=1e-4):
    # 1. Load the CSV files
    try:
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. Basic Structure Check
    if df1.shape != df2.shape:
        print(f"Structure Mismatch: File 1 is {df1.shape}, File 2 is {df2.shape}")
        return
    
    # Check if column names match
    if list(df1.columns) != list(df2.columns):
        print("Column names do not match.")
        return

    print(f"Comparing files with shape {df1.shape}...")

    # 3. Separate Numeric and Non-Numeric Data
    # We only want to check tolerance on numbers. Strings must match exactly.
    numeric_cols = df1.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df1.select_dtypes(exclude=[np.number]).columns

    # 4. Compare Non-Numeric Columns (Exact Match)
    if not df1[non_numeric_cols].equals(df2[non_numeric_cols]):
        print("Differences found in non-numeric (text) columns:")
        diff_text = df1[non_numeric_cols].compare(df2[non_numeric_cols])
        print(diff_text)
    else:
        print("Text/String columns match exactly.")

    # 5. Compare Numeric Columns (With Tolerance)
    # This subtracts the two dataframes and checks if the absolute difference is > tolerance
    numeric_diff = np.abs(df1[numeric_cols] - df2[numeric_cols])
    
    # Create a boolean mask of values exceeding tolerance
    exceeds_tolerance = numeric_diff > tolerance

    if exceeds_tolerance.any().any():
        print(f"\nDifferences larger than {tolerance} found in numeric columns:")
        
        # Show specific counts of errors per column
        for col in numeric_cols:
            col_mask = exceeds_tolerance[col]
            count = col_mask.sum()
            if count > 0:
                max_diff = numeric_diff[col].max()
                print(f" - Column '{col}': {count} discrepancies (Max difference: {max_diff})")
            # Extract the specific rows where the difference exists
                # We create a new dataframe just to display the side-by-side values
                diff_view = pd.DataFrame({
                    'df1_value': df1.loc[col_mask, col],
                    'df2_value': df2.loc[col_mask, col],
                    'difference': numeric_diff.loc[col_mask, col]
                })
                # Print the values (defaulting to showing the first 10 for brevity)
                print(diff_view.head(30).to_string())
        
    else:
        print(f"\nSuccess! No numeric differences larger than {tolerance} were found.")

# --- Usage ---
# Replace these filenames with your actual local paths if they differ
file1 = 'cifar10_sam-.csv'
file2 = 'cifar10_samm.csv'

compare_csv_files(file1, file2)