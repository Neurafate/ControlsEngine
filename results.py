import pandas as pd

def merge_results_with_framework1(original_framework1_path, comparison_results_path, output_path):
    """
    Merges the comparison results back into the original framework 1 dataset.
    Saves the merged result as a new CSV file.
    
    Parameters:
    - original_framework1_path: Path to the original framework 1 CSV before classification.
    - comparison_results_path: Path to the CSV file with the comparison results.
    - output_path: Path where the merged CSV will be saved.
    """
    # Step 1: Load the original framework 1 CSV
    framework1_df = pd.read_csv(original_framework1_path)

    # Step 2: Load the comparison results CSV
    comparison_results_df = pd.read_csv(comparison_results_path)

    # Step 3: Merge the results into the original framework 1 DataFrame
    # Use 'Control from F1' from comparison results to match the 'Requirement' column in framework 1
    merged_df = framework1_df.merge(
        comparison_results_df[['Control from F1', 'Best Match from F2', 'Similarity Score', 'Match Type']],
        left_on='Requirement', right_on='Control from F1',
        how='left'
    )

    # Step 5: Save the merged DataFrame to a new CSV file
    merged_df.to_csv(output_path, index=False)

    print(f"Merged results saved to: {output_path}")

if __name__ == '__main__':
    # Define file paths
    original_framework1_path = 'Original.csv'  # Path to original framework 1 CSV (before
    comparison_results_path = 'control_comparisons.csv'   # Path to the comparison results CSV
    output_path = 'framework1_with_results.csv'           # Output path for the merged CSV

    # Run the merge process
    merge_results_with_framework1(original_framework1_path, comparison_results_path, output_path)
