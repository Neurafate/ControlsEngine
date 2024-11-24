from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from thefuzz import fuzz
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import time

app = Flask(__name__)
CORS(app)

# Download necessary NLTK data files
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize Sentence Transformer
model = SentenceTransformer('all-mpnet-base-v2')

# Initialize Lemmatizer and Stop Words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Directory to save uploaded files
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def preprocess_text(text):
    if pd.isnull(text):
        return ''
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

@app.route('/process', methods=['POST'])
def process_files():
    start_time = time.time()
    print("Received request to process files")

    # Get uploaded files and Top-K value
    file1 = request.files.get('frame1')
    file2 = request.files.get('frame2')
    top_k = int(request.form.get('top_k', 5))
    print(f"Top-K value set to: {top_k}")

    if not file1 or not file2:
        print("Error: Both files are required")
        return jsonify({"error": "Both files are required"}), 400

    # Save uploaded files
    path1 = os.path.join(UPLOAD_FOLDER, file1.filename)
    path2 = os.path.join(UPLOAD_FOLDER, file2.filename)
    file1.save(path1)
    file2.save(path2)
    print(f"Files saved: {path1}, {path2}")

    # Load the Excel files
    print("Loading Excel files...")
    df1 = pd.read_excel(path1)
    df2 = pd.read_excel(path2)
    print(f"Loaded df1 with shape {df1.shape}")
    print(f"Loaded df2 with shape {df2.shape}")

    # Forward-fill missing 'Domain' and 'Sub-Domain' values
    df1['Domain'] = df1['Domain'].fillna(method='ffill')
    df2['Domain'] = df2['Domain'].fillna(method='ffill')
    df1['Sub-Domain'] = df1['Sub-Domain'].fillna(method='ffill')
    df2['Sub-Domain'] = df2['Sub-Domain'].fillna(method='ffill')
    print("Forward-filled missing values for 'Domain' and 'Sub-Domain'")

    # Combine and preprocess text columns
    print("Preprocessing text columns...")
    df1['Combined_Text'] = df1['Domain'].astype(str) + ' ' + df1['Sub-Domain'].astype(str) + ' ' + df1['Control'].astype(str)
    df2['Combined_Text'] = df2['Domain'].astype(str) + ' ' + df2['Sub-Domain'].astype(str) + ' ' + df2['Control'].astype(str)
    df1['Processed_Control'] = df1['Combined_Text'].apply(preprocess_text)
    df2['Processed_Control'] = df2['Combined_Text'].apply(preprocess_text)
    print("Text preprocessing completed")

    # Semantic and TF-IDF similarity
    results = []
    tfidf_weight = 0.4
    embedding_weight = 0.6
    control_threshold = 0.35

    print("Starting domain-wise similarity computation...")
    for domain in df1['Domain'].unique():
        print(f"Processing domain: {domain}")
        controls_f1 = df1[df1['Domain'] == domain].reset_index(drop=True)
        controls_f2 = df2[df2['Domain'] == domain].reset_index(drop=True)

        if controls_f1.empty or controls_f2.empty:
            print(f"No matching data for domain: {domain}")
            continue

        texts_f1 = controls_f1['Processed_Control'].tolist()
        texts_f2 = controls_f2['Processed_Control'].tolist()

        # Embedding Similarity
        print(f"Computing embeddings for domain: {domain}")
        start_embeddings = time.time()
        embeddings_f1 = model.encode(texts_f1, convert_to_tensor=True)
        embeddings_f2 = model.encode(texts_f2, convert_to_tensor=True)
        print(f"Embedding computation took {time.time() - start_embeddings:.2f} seconds")

        embedding_similarity = util.cos_sim(embeddings_f1, embeddings_f2).cpu().numpy()

        # TF-IDF Similarity
        print(f"Computing TF-IDF for domain: {domain}")
        start_tfidf = time.time()
        vectorizer = TfidfVectorizer().fit(texts_f1 + texts_f2)
        tfidf_f1 = vectorizer.transform(texts_f1)
        tfidf_f2 = vectorizer.transform(texts_f2)
        print(f"TF-IDF computation took {time.time() - start_tfidf:.2f} seconds")

        tfidf_similarity = cosine_similarity(tfidf_f1, tfidf_f2)

        # Combined Similarity
        combined_similarity = (embedding_similarity * embedding_weight +
                               tfidf_similarity * tfidf_weight) / (embedding_weight + tfidf_weight)

        for idx_f1, row_f1 in controls_f1.iterrows():
            similarities = combined_similarity[idx_f1]
            top_indices = similarities.argsort()[-top_k:][::-1]
            top_scores = similarities[top_indices]
            matching_indices = [i for i, score in zip(top_indices, top_scores) if score >= control_threshold]
            matching_controls = [controls_f2.iloc[i]['Control'] for i in matching_indices]
            matching_scores = [similarities[i] for i in matching_indices]

            if not matching_controls:
                matching_controls = [None]
                matching_scores = [None]

            results.append({
                'Domain': domain,
                'Sub-Domain': row_f1['Sub-Domain'],
                'Control': row_f1['Control'],
                'Controls from F2 that match': matching_controls,
                'Similarity Scores': matching_scores
            })
        print(f"Processed {len(controls_f1)} controls for domain: {domain}")

    print("Domain-wise similarity computation completed")

    # Create and save the output DataFrame
    print("Creating final DataFrame...")
    final_df = pd.DataFrame(results)
    final_df = final_df.explode(['Controls from F2 that match', 'Similarity Scores']).reset_index(drop=True)
    final_df['Similarity Scores'] = final_df['Similarity Scores'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else '')
    output_path = os.path.join(OUTPUT_FOLDER, "framework1_with_results.csv")
    final_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds")

    return jsonify({"data": final_df.to_dict(orient='records'), "processing_time": f"{total_time:.2f} seconds"}), 200

@app.route('/download/framework1_with_results.csv', methods=['GET'])
def download_file():
    output_path = os.path.join(OUTPUT_FOLDER, "framework1_with_results.csv")
    if not os.path.exists(output_path):
        print("Error: File not found for download")
        return jsonify({"error": "File not found"}), 404
    print(f"File downloaded: {output_path}")
    return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
