import os
import pandas as pd
import pickle
import csv

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

if __name__ == '__main__':
    # Example usage:
    # Replace these paths with the actual file paths of your two uploaded files
    file1 = 'test1.xlsx'  # or .xlsx
    file2 = 'test2.xlsx'  # or .xlsx

    # Process the files
    try:
        classified_files = process_files([file1, file2])
        print(f"Classified files saved at: {classified_files}")
    except Exception as e:
        print(f"Error occurred: {e}")