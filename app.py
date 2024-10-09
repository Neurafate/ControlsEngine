from flask import Flask, request, send_file, jsonify, url_for
import time
import pandas as pd
import os
import pickle
import csv
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Define the absolute paths to the model and vectorizer files
BASE_DIR = app.root_path
MODEL_PATH = os.path.join(BASE_DIR, 'model_weights.pkl')
VECTOR_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')

# Load the model and vectorizer
with open(MODEL_PATH, 'rb') as f:
    classification_model = pickle.load(f)

with open(VECTOR_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

def process(input_files):
    dfs = []
    for idx, input_file in enumerate(input_files, start=1):
        # Read the Excel file
        df = pd.read_excel(input_file)
        
        # Select only the third column (column C) and rename it to 'control'
        df_control = df.iloc[:, [2]].copy()  # iloc[:, 2] selects the third column
        df_control.columns = ['control']     # Rename it to 'control'
        
        # Add the resulting DataFrame to the list
        dfs.append(df_control)
        
    # Return the list of DataFrames
    return dfs

def predict_category(control):
    """Predict the category for a given control value."""
    combined_features = f"{control}"
    control_tfidf = vectorizer.transform([combined_features])
    return classification_model.predict(control_tfidf)[0]

def classify_df(df):
    """Classify the 'control' column of the given DataFrame."""
    
    # Ensure the 'control' column is present
    if 'control' not in df.columns:
        raise ValueError(f'Missing "control" column in the DataFrame')

    # Apply classification
    df['predicted_label'] = df['control'].apply(predict_category)  # Assuming predict_category is a predefined function

    return df  # Return the classified DataFrame

def process_dfs(dfs):
    """Process and classify two DataFrames."""
    if len(dfs) != 2:
        raise ValueError("Two DataFrames are required for processing.")

    classified_dfs = []
    for df in dfs:
        try:
            classified_df = classify_df(df)  # Assuming classify_df is the function to classify a DataFrame
            classified_dfs.append(classified_df)
        except Exception as e:
            print(f"Error processing DataFrame: {e}")

    return classified_dfs

def group_controls_by_label(df1, df2):
    """
    Load and group controls by their predicted labels from two DataFrames.
    Returns two dictionaries with controls grouped by labels.
    """

    # Check for the required columns in both DataFrames
    required_columns = ['predicted_label', 'control']
    if not all(col in df1.columns for col in required_columns):
        raise ValueError(f"Missing required columns in DataFrame 1. Expected columns: {required_columns}")
    if not all(col in df2.columns for col in required_columns):
        raise ValueError(f"Missing required columns in DataFrame 2. Expected columns: {required_columns}")

    # Initialize dictionaries to store controls grouped by labels
    grouped_controls_df1 = {}
    grouped_controls_df2 = {}

    # Group controls in df1 by their labels
    for _, row in df1.iterrows():
        label = row['predicted_label']
        control = row['control']
        
        if label not in grouped_controls_df1:
            grouped_controls_df1[label] = []
        grouped_controls_df1[label].append(control)

    # Group controls in df2 by their labels
    for _, row in df2.iterrows():
        label = row['predicted_label']
        control = row['control']
        
        if label not in grouped_controls_df2:
            grouped_controls_df2[label] = []
        grouped_controls_df2[label].append(control)

    # Return the two dictionaries
    return grouped_controls_df1, grouped_controls_df2

def compute_embeddings(controls, embedding_model):
    """
    Generate embeddings for a list of controls using a pre-trained model.
    """
    embeddings = embedding_model.encode(controls, convert_to_tensor=True)
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
                'Control from User Org Framework': controls1[i],
                'Best Match from Service Org Framework': best_match_control2,
                'Similarity Score': best_match_score,
                'Match Type': match_type
            })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def run_comparison(df1, df2):
    """
    Main function to compare controls from two DataFrames and store the results
    as a DataFrame.
    """
    # Step 1: Group controls by label
    grouped_controls_df1, grouped_controls_df2 = group_controls_by_label(df1, df2)  # Assuming group_controls_by_label works with DataFrames

    # Load a pre-trained model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    all_results_df = pd.DataFrame()

    # Step 2: Compare controls for each matching label between the two DataFrames
    for label in grouped_controls_df1:
        if label in grouped_controls_df2:
            controls1 = grouped_controls_df1[label]
            controls2 = grouped_controls_df2[label]

            # Step 3: Compute embeddings for both sets of controls
            embeddings1 = compute_embeddings(controls1, embedding_model)  # Assuming compute_embeddings is defined
            embeddings2 = compute_embeddings(controls2, embedding_model)

            # Step 4: Compare embeddings and get results DataFrame
            results_df = compare_controls(controls1, embeddings1, controls2, embeddings2)  # Assuming compare_controls is defined

            # Append to overall results
            all_results_df = pd.concat([all_results_df, results_df], ignore_index=True)

    # Step 5: Optionally, show the DataFrame output or return it instead of saving to a CSV
    print("Comparison complete.")
    
    return all_results_df  # Return the DataFrame for further use

def run_comparison_faiss(df1, df2, embedding_model):
    """
    Function to compare controls using FAISS for nearest neighbor search instead of SBERT cosine similarity.
    """
    # Step 1: Group controls by label
    grouped_controls_df1, grouped_controls_df2 = group_controls_by_label(df1, df2)

    all_results_df = pd.DataFrame()
    results=pd.DataFrame()
    # Step 2: Compare controls for each matching label between the two DataFrames
    for label in grouped_controls_df1:
        if label in grouped_controls_df2:
            controls1 = grouped_controls_df1[label]
            controls2 = grouped_controls_df2[label]

            # Step 3: Compute embeddings for both sets of controls using SBERT or another model
            embeddings1 = compute_embeddings(controls1, embedding_model)
            embeddings2 = compute_embeddings(controls2, embedding_model)

            # Step 4: Set up FAISS index
            dim = embeddings1.shape[1]  # Assuming embeddings have the same dimension
            index = faiss.IndexFlatL2(dim)  # L2 distance (Euclidean)

            # Step 5: Add embeddings from controls2 to the FAISS index
            index.add(embeddings2.cpu().detach().numpy())  # Assuming embeddings2 is a tensor; convert to numpy

            # Step 6: Perform search for the nearest neighbor for each control in controls1
            D, I = index.search(embeddings1.cpu().detach().numpy(), 1)  # Search for 1 nearest neighbor

            # Step 7: Collect results based on FAISS results
            for i, control1 in enumerate(controls1):
                best_match_index = I[i][0]  # Index of the best match in controls2
                best_match_score = 1 - (D[i][0] / 2)  # Convert L2 distance to similarity score
                
                match_type = 'No Match'
                if best_match_score >= 0.8:
                    match_type = 'Full Match'
                elif best_match_score >= 0.5:
                    match_type = 'Partial Match'

                results.append({
                    'Control from User Org Framework': control1,
                    'Best Match from Service Org Framework': controls2[best_match_index],
                    'Similarity Score': best_match_score,
                    'Match Type': match_type
                })

            # Append to overall results
            all_results_df = pd.concat([all_results_df, pd.DataFrame(results)], ignore_index=True)

    # Step 8: Return the final DataFrame with comparison results
    return all_results_df

def merge_results_with_framework1(original_framework1_path, comparison_results_path, output_path):
    """
    Merges the comparison results back into the original framework 1 dataset.
    Saves the merged result as a new CSV file.
    """
    # Step 1: Load the original framework 1 CSV
    framework1_df = original_framework1_path

    # Step 2: Load the comparison results CSV
    comparison_results_df = comparison_results_path

    # Step 3: Merge the results into the original framework 1 DataFrame
    # Use 'Control from F1' from comparison results to match the 'Requirement' column in framework 1
    merged_df = framework1_df.merge(
        comparison_results_df[['Control from User Org Framework', 'Best Match from Service Org Framework', 'Similarity Score', 'Match Type']],
        left_on='Requirement', right_on='Control from User Org Framework',
        how='left'
    )
    merged_df = merged_df.drop(columns=['Control from User Org Framework'])

    # Step 5: Save the merged DataFrame
    merged_df.to_csv(output_path, index=False)
    return merged_df

@app.route('/process', methods=['POST'])
def process_files_endpoint():
    if request.method == 'POST':
        start_time = time.time()  # Start time

        # Check if the files are part of the request
        if 'frame1' not in request.files or 'frame2' not in request.files:
            return jsonify({'error': 'Both "frame1" and "frame2" files are required.'}), 400
        file1 = request.files['frame1']
        file2 = request.files['frame2']
        # Get user choice from the form data
#        user_choice = int(request.form.get('faissChoice', 0))  # Default is 0 for not using FAISS

        # If user does not select file, browser may submit an empty part without filename
        if file1.filename == '' or file2.filename == '':
            return jsonify({'error': 'No file selected for uploading.'}), 400
        if file1 and file2:
            try:
                # Save uploaded files to disk
                file1_path = os.path.join(BASE_DIR, 'uploaded_frame1.xlsx')
                file2_path = os.path.join(BASE_DIR, 'uploaded_frame2.xlsx')
                file1.save(file1_path)
                file2.save(file2_path)
                # Process the files
                frameworks = [file1_path, file2_path]
                test_DFs = process(frameworks)

                # Process files and classify
                classified_DFs = process_dfs(test_DFs)

                # Run comparison
                classified_DF1 = classified_DFs[0]
                classified_DF2 = classified_DFs[1]

                user_choice = 0  # Default is 0 for not using FAISS
                if user_choice == 0:
                    Compared_DF = run_comparison(classified_DF1, classified_DF2)
                else:
                    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    Compared_DF = run_comparison_faiss(classified_DF1, classified_DF2, embedding_model)

                # Save original frame1 as original.csv
                Original_DF = pd.read_excel(file1_path)

                # Merge results
                output_csv_path = os.path.join(BASE_DIR, 'framework1_with_results.csv')
                merged_df = merge_results_with_framework1(Original_DF, Compared_DF, output_csv_path)
                response_json = {
                    'data': merged_df.to_json(orient='records'),
                    'download_url': url_for('download_csv', filename='framework1_with_results.csv')
                }

                end_time = time.time()  # End time
                processing_time = end_time - start_time  # Calculate time taken
                print(processing_time)
                response_json['processing_time'] = f"{processing_time:.2f} seconds"  # Add time to response

                # Send the final merged file to the user
                return jsonify(response_json)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'Invalid files uploaded.'}), 400
        
@app.route('/download/<filename>', methods=['GET'])
def download_csv(filename):
    """Allow users to download the processed CSV file."""
    file_path = os.path.join(BASE_DIR, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({'error': 'File not found.'}), 404       

if __name__ == '__main__':
    app.run(debug=True)