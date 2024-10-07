from flask import Flask, request, jsonify, send_file
import os
import time
import pandas as pd
import pickle
import csv
from sentence_transformers import SentenceTransformer, util
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define constants for paths
BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, 'model_weights.pkl')
VECTOR_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')

# Load the model and vectorizer
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(VECTOR_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

# Helper functions
def process(input_files):
    for idx, input_file in enumerate(input_files, start=1):
        df = pd.read_excel(input_file)
        df_control = df.iloc[:, [2]].copy()
        df_control.columns = ['control']
        output_file = f'test{idx}.xlsx'
        df_control.to_excel(output_file, index=False)

def predict_category(control):
    combined_features = f"{control}"
    control_tfidf = vectorizer.transform([combined_features])
    return model.predict(control_tfidf)[0]

def classify_file(file_path):
    if file_path.lower().endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.lower().endswith('.xlsx'):
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")

    if 'control' not in data.columns:
        raise ValueError(f'Missing "control" column in the file {file_path}')

    data['predicted_label'] = data['control'].apply(predict_category)
    csv_filename = f'classified_{os.path.basename(file_path).rsplit(".", 1)[0]}.csv'
    csv_file_path = os.path.join(BASE_DIR, csv_filename)
    data.to_csv(csv_file_path, index=False, quoting=csv.QUOTE_MINIMAL)

    return csv_file_path

def process_files(file_paths):
    if len(file_paths) != 2:
        raise ValueError("Two files are required for processing.")
    
    classified_files = []
    for file_path in file_paths:
        classified_file = classify_file(file_path)
        classified_files.append(classified_file)
    
    return classified_files

def run_comparison(sheet1_path, sheet2_path):
    # Comparison logic here...
    pass  # Implementation as in your original code

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files[]')
    if len(files) != 2:
        return jsonify({'error': 'Please upload exactly two files'}), 400

    saved_files = []
    for file in files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(BASE_DIR, filename)
        file.save(file_path)
        saved_files.append(file_path)

    try:
        process(saved_files)
        classified_files = process_files([f'test1.xlsx', f'test2.xlsx'])
        run_comparison(sheet1_path=classified_files[0], sheet2_path=classified_files[1])
        return send_file('control_comparisons.csv', as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/merge', methods=['POST'])
def merge_files():
    if 'framework1' not in request.files or 'comparison' not in request.files:
        return jsonify({'error': 'Missing required files'}), 400

    framework1 = request.files['framework1']
    comparison = request.files['comparison']

    framework1_path = secure_filename(framework1.filename)
    comparison_path = secure_filename(comparison.filename)

    framework1.save(framework1_path)
    comparison.save(comparison_path)

    output_path = 'framework1_with_results.csv'
    merge_results_with_framework1(framework1_path, comparison_path, output_path)

    return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)