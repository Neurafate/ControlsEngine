# app.py
import os
import time
import pandas as pd
import pickle
import csv
from flask import Flask, request, render_template, redirect, url_for, send_file, flash
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your own secret key
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define the absolute paths to the model and vectorizer files
BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, 'model_weights.pkl')
VECTOR_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')

# Load the model and vectorizer
with open(MODEL_PATH, 'rb') as f:
    model_classifier = pickle.load(f)

with open(VECTOR_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

# Load the SentenceTransformer model once
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process(input_files, output_dir):
    processed_files = []
    for idx, input_file in enumerate(input_files, start=1):
        # Read the Excel or CSV file
        if input_file.lower().endswith('.csv'):
            df = pd.read_csv(input_file)
        else:
            df = pd.read_excel(input_file)
        
        # Select only the third column (column C) and rename it to 'control'
        df_control = df.iloc[:, [2]].copy()  # iloc[:, 2] selects the third column
        df_control.columns = ['control']     # Rename it to 'control'
        
        # Create a unique output file name (test1.xlsx, test2.xlsx, etc.)
        output_file = os.path.join(output_dir, f'test{idx}.xlsx')
        
        # Save the result to a new Excel file
        df_control.to_excel(output_file, index=False)
        processed_files.append(output_file)
    return processed_files

def predict_category(control):
    """Predict the category for a given control value."""
    combined_features = f"{control}"
    control_tfidf = vectorizer.transform([combined_features])
    return model_classifier.predict(control_tfidf)[0]

def classify_file(file_path, output_dir):
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
    csv_file_path = os.path.join(output_dir, csv_filename)
    data.to_csv(csv_file_path, index=False, quoting=csv.QUOTE_MINIMAL)

    return csv_file_path

def process_files(file_paths, output_dir):
    """Process and classify two uploaded files."""
    if len(file_paths) != 2:
        raise ValueError("Two files are required for processing.")

    classified_files = []
    for file_path in file_paths:
        try:
            classified_file = classify_file(file_path, output_dir)
            classified_files.append(classified_file)
        except Exception as e:
            raise RuntimeError(f"Error processing {file_path}: {e}")

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
                break  # Assuming we take the first full match
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

def run_comparison(sheet1_path, sheet2_path, output_dir):
    """
    Main function to compare controls from two CSV files and store the results
    as a DataFrame.
    """
    # Step 1: Group controls by label
    grouped_controls_sheet1, grouped_controls_sheet2 = group_controls_by_label(sheet1_path, sheet2_path)

    all_results_df = pd.DataFrame()

    # Step 2: Compare controls for each matching label between the two sheets
    for label in grouped_controls_sheet1:
        if label in grouped_controls_sheet2:
            controls1 = grouped_controls_sheet1[label]
            controls2 = grouped_controls_sheet2[label]

            # Step 3: Compute embeddings for both sets of controls
            embeddings1 = compute_embeddings(controls1, sentence_model)
            embeddings2 = compute_embeddings(controls2, sentence_model)

            # Step 4: Compare embeddings and get results DataFrame
            results_df = compare_controls(controls1, embeddings1, controls2, embeddings2)

            # Append to overall results
            all_results_df = pd.concat([all_results_df, results_df], ignore_index=True)

    # Step 5: Save results as a CSV file
    comparison_csv_path = os.path.join(output_dir, 'control_comparisons.csv')
    all_results_df.to_csv(comparison_csv_path, index=False)
    print("Comparison complete. Results saved to 'control_comparisons.csv'.")

    return comparison_csv_path

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
    if original_framework1_path.lower().endswith('.csv'):
        framework1_df = pd.read_csv(original_framework1_path)
    else:
        framework1_df = pd.read_excel(original_framework1_path)

    # Step 2: Load the comparison results CSV
    comparison_results_df = pd.read_csv(comparison_results_path)

    # Step 3: Merge the results into the original framework 1 DataFrame
    # Use 'Control from F1' from comparison results to match the 'Requirement' column in framework 1
    merged_df = framework1_df.merge(
        comparison_results_df[['Control from F1', 'Best Match from F2', 'Similarity Score', 'Match Type']],
        left_on='Requirement', right_on='Control from F1',
        how='left'
    )

    # Optionally, drop the redundant 'Control from F1' column
    merged_df.drop(columns=['Control from F1'], inplace=True)

    # Step 4: Save the merged DataFrame to a new CSV file
    merged_df.to_csv(output_path, index=False)

    print(f"Merged results saved to: {output_path}")

    return output_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_files_route():
    if 'frame1' not in request.files or 'frame2' not in request.files:
        flash('Both files are required.')
        return redirect(request.url)

    file1 = request.files['frame1']
    file2 = request.files['frame2']

    if file1.filename == '' or file2.filename == '':
        flash('No selected files.')
        return redirect(request.url)

    if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename):
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)
        file1_path = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        file2_path = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        file1.save(file1_path)
        file2.save(file2_path)

        try:
            # Create a unique output directory based on timestamp
            timestamp = int(time.time())
            output_dir = os.path.join('outputs', f'output_{timestamp}')
            os.makedirs(output_dir, exist_ok=True)

            # Step 1: Process the uploaded files to extract 'control' column
            processed_files = process([file1_path, file2_path], output_dir)

            # Step 2: Classify the processed files
            classified_files = process_files(processed_files, output_dir)

            # Step 3: Run comparison
            comparison_csv = run_comparison(classified_files[0], classified_files[1], output_dir)

            # Step 4: Convert original framework1 to CSV if it's not already
            original_csv_path = os.path.join(output_dir, 'original.csv')
            if file1_path.lower().endswith('.csv'):
                original = file1_path
            else:
                df_original = pd.read_excel(file1_path)
                df_original.to_csv(original_csv_path, index=False)
                original = original_csv_path

            # Step 5: Merge results
            merged_output_path = os.path.join(output_dir, 'framework1_with_results.csv')
            merge_results_with_framework1(original, comparison_csv, merged_output_path)

            # Provide the merged file for download
            return send_file(merged_output_path, as_attachment=True)

        except Exception as e:
            flash(f'An error occurred during processing: {e}')
            return redirect(url_for('index'))
    else:
        flash('Allowed file types are .xlsx and .csv')
        return redirect(request.url)

if __name__ == '__main__':
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    app.run(debug=True)
