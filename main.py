import time
import pandas as pd
import os
import pickle
import csv
from sentence_transformers import SentenceTransformer, util
from results import merge_results_with_framework1

def process(input_files):
    for idx, input_file in enumerate(input_files, start=1):
            # Read the Excel file
            df = pd.read_excel(input_file)
            
            # Select only the third column (column C) and rename it to 'control'
            df_control = df.iloc[:, [2]].copy()  # iloc[:, 2] selects the third column
            df_control.columns = ['control']     # Rename it to 'control'
            
            # Create a unique output file name (test1.xlsx, test2.xlsx, etc.)
            output_file = f'test{idx}.xlsx'
            
            # Save the result to a new Excel file
            df_control.to_excel(output_file, index=False)

# Define the absolute paths to the model and vectorizer files
BASE_DIR = os.getcwd()  # Use os.getcwd() for environments where __file__ is not defined
MODEL_PATH = os.path.join(BASE_DIR, 'model_weights.pkl')
VECTOR_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')

# Load the model and vectorizer
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(VECTOR_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

def predict_category(control):
    """Predict the category for a given control value."""
    combined_features = f"{control}"
    control_tfidf = vectorizer.transform([combined_features])
    return model.predict(control_tfidf)[0]

def classify_file(file_path):
    """Classify the 'control' column of the uploaded file."""
    # Read file based on extension
    if file_path.lower().endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.lower().endswith('.xlsx'):
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")

    # Ensure the 'control' column is present
    if 'control' not in data.columns:
        raise ValueError(f'Missing "control" column in the file {file_path}')

    # Apply classification
    data['predicted_label'] = data['control'].apply(predict_category)

    # Save classified data as CSV
    csv_filename = f'classified_{os.path.basename(file_path).rsplit(".", 1)[0]}.csv'
    csv_file_path = os.path.join(BASE_DIR, csv_filename)
    data.to_csv(csv_file_path, index=False, quoting=csv.QUOTE_MINIMAL)

    return csv_file_path

def process_files(file_paths):
    """Process and classify two uploaded files."""
    if len(file_paths) != 2:
        raise ValueError("Two files are required for processing.")

    classified_files = []
    for file_path in file_paths:
        try:
            classified_file = classify_file(file_path)
            classified_files.append(classified_file)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return classified_files

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

frame1=str(input("Enter the path of user org framework"))
frame2=str(input("Enter the path of the service org framework"))
frameworks=[frame1,frame2]
process(frameworks)

file1='test1.xlsx'
file2='test2.xlsx'

time.sleep(10)
classifiedfiles=process_files([file1,file2])

sheet1=classifiedfiles[0]
sheet2=classifiedfiles[1]
run_comparison(sheet1_path=sheet1,sheet2_path=sheet2)

df = pd.read_excel(frame1)
csv_file_path = 'original.csv'  # Replace with the desired output file path
df.to_csv(csv_file_path, index=False)

original='original.csv'
comparison='control_comparisons.csv'
output_path = 'framework1_with_results.csv'
merge_results_with_framework1(original, comparison, output_path)