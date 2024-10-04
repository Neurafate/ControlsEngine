import pandas as pd
from sentence_transformers import SentenceTransformer, util

def group_controls_by_label(sheet1_path, sheet2_path):
    """
    Load and group controls by their predicted labels from two CSV files.
    Returns two dictionaries with controls grouped by labels.
    """
    # Load the CSV files into pandas DataFrames
    sheet1 = pd.read_csv(sheet1_path)
    sheet2 = pd.read_csv(sheet2_path)

    # Check for the required columns in both sheets
    required_columns = ['predicted_label', 'control']
    if not all(col in sheet1.columns for col in required_columns):
        raise ValueError(f"Missing required columns in sheet1. Expected columns: {required_columns}")
    if not all(col in sheet2.columns for col in required_columns):
        raise ValueError(f"Missing required columns in sheet2. Expected columns: {required_columns}")

    # Initialize dictionaries to store controls grouped by labels
    grouped_controls_sheet1 = {}
    grouped_controls_sheet2 = {}

    # Group controls in sheet1 by their labels
    for _, row in sheet1.iterrows():
        label = row['predicted_label']
        control = row['control']
        
        if label not in grouped_controls_sheet1:
            grouped_controls_sheet1[label] = []
        grouped_controls_sheet1[label].append(control)

    # Group controls in sheet2 by their labels
    for _, row in sheet2.iterrows():
        label = row['predicted_label']
        control = row['control']
        
        if label not in grouped_controls_sheet2:
            grouped_controls_sheet2[label] = []
        grouped_controls_sheet2[label].append(control)

    # Return the two dictionaries
    return grouped_controls_sheet1, grouped_controls_sheet2

def compute_embeddings(controls, model):
    """
    Generate embeddings for a list of controls using a pre-trained model.
    """
    embeddings = model.encode(controls, convert_to_tensor=True)
    return embeddings

def compare_controls(controls1, embeddings1, controls2, embeddings2, threshold_full=0.8, threshold_partial=0.5):
    """
    Compare the controls from two different frameworks based on cosine similarity
    of their embeddings. Returns a DataFrame showing matched controls.
    """
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    results = []

    for i in range(len(controls1)):
        best_match_score = -1
        best_match_control2 = None
        match_type = 'No Match'

        for j in range(len(controls2)):
            score = cosine_scores[i][j].item()

            # Find the best match for control1[i] in controls2
            if score >= threshold_full:
                best_match_score = score
                best_match_control2 = controls2[j]
                match_type = 'Full Match'
            elif score >= threshold_partial and score > best_match_score:
                best_match_score = score
                best_match_control2 = controls2[j]
                match_type = 'Partial Match'

        if best_match_control2:
            results.append({
                'Control from F1': controls1[i],
                'Best Match from F2': best_match_control2,
                'Similarity Score': best_match_score,
                'Match Type': match_type
            })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def run_comparison(sheet1_path, sheet2_path):
    """
    Main function to compare controls from two CSV files and store the results
    as a DataFrame.
    """
    # Step 1: Group controls by label
    grouped_controls_sheet1, grouped_controls_sheet2 = group_controls_by_label(sheet1_path, sheet2_path)

    # Load a pre-trained model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    all_results_df = pd.DataFrame()

    # Step 2: Compare controls for each matching label between the two sheets
    for label in grouped_controls_sheet1:
        if label in grouped_controls_sheet2:
            controls1 = grouped_controls_sheet1[label]
            controls2 = grouped_controls_sheet2[label]

            # Step 3: Compute embeddings for both sets of controls
            embeddings1 = compute_embeddings(controls1, model)
            embeddings2 = compute_embeddings(controls2, model)

            # Step 4: Compare embeddings and get results DataFrame
            results_df = compare_controls(controls1, embeddings1, controls2, embeddings2)

            # Append to overall results
            all_results_df = pd.concat([all_results_df, results_df], ignore_index=True)

    # Step 5: Save results as a CSV file
    all_results_df.to_csv('control_comparisons.csv', index=False)
    print("Comparison complete. Results saved to 'control_comparisons.csv'.")

    # Step 6: Optionally, show the DataFrame output
    print(all_results_df)

if __name__ == '__main__':
    # Define paths to your classified control CSV files
    sheet1_path = 'classified_test1.csv'
    sheet2_path = 'classified_test2.csv'
    
    # Run the comparison process
    run_comparison(sheet1_path, sheet2_path)