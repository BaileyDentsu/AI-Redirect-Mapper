import os
import chardet
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename
from polyfuzz import PolyFuzz
from polyfuzz.models import TFIDF, EditDistance, RapidFuzz
from sentence_transformers import SentenceTransformer
import faiss

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'  # Change this to a secure key

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Utility Functions -----------------------------------------------------------------------------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_csv_with_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    return pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')

def read_file(file_path):
    if file_path.endswith('.csv'):
        return read_csv_with_encoding(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return pd.read_excel(file_path)
    else:
        return None

def get_sbert_embeddings(text_list):
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = sbert_model.encode(text_list, show_progress_bar=True)
    return embeddings

def match_using_model(model, live_list, staging_list):
    if model == "SBERT & Cosine Similarity":
        live_embeddings = get_sbert_embeddings(live_list)
        staging_embeddings = get_sbert_embeddings(staging_list)

        # Normalize embeddings
        live_embeddings_norm = live_embeddings / np.linalg.norm(live_embeddings, axis=1, keepdims=True)
        staging_embeddings_norm = staging_embeddings / np.linalg.norm(staging_embeddings, axis=1, keepdims=True)

        cosine_similarities = np.dot(live_embeddings_norm, staging_embeddings_norm.T)

        matches = []
        for idx in range(len(live_list)):
            max_idx = cosine_similarities[idx].argmax()
            max_score = cosine_similarities[idx][max_idx]
            matches.append((live_list[idx], staging_list[max_idx], max_score))
        return matches
    else:
        model.match(live_list, staging_list)
        matches = model.get_matches()
        return matches

def setup_matching_model(selected_model):
    if selected_model == "Edit Distance":
        return PolyFuzz(EditDistance())
    elif selected_model == "RapidFuzz":
        return PolyFuzz(RapidFuzz())
    elif selected_model == "SBERT & Cosine Similarity":
        return "SBERT & Cosine Similarity"
    else:
        return PolyFuzz(TFIDF())

# Routes -----------------------------------------------------------------------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Upload files
        file_live = request.files['file_live']
        file_staging = request.files['file_staging']

        if file_live and file_staging and allowed_file(file_live.filename) and allowed_file(file_staging.filename):
            filename_live = secure_filename(file_live.filename)
            filename_staging = secure_filename(file_staging.filename)

            path_live = os.path.join(app.config['UPLOAD_FOLDER'], filename_live)
            path_staging = os.path.join(app.config['UPLOAD_FOLDER'], filename_staging)

            file_live.save(path_live)
            file_staging.save(path_staging)

            return redirect(url_for('process_files', live=filename_live, staging=filename_staging))

    return render_template('index.html')

@app.route('/process/<live>/<staging>')
def process_files(live, staging):
    path_live = os.path.join(app.config['UPLOAD_FOLDER'], live)
    path_staging = os.path.join(app.config['UPLOAD_FOLDER'], staging)

    df_live = read_file(path_live)
    df_staging = read_file(path_staging)

    if df_live is None or df_staging is None:
        flash("Invalid file type.")
        return redirect(url_for('index'))

    model = setup_matching_model("TF-IDF")

    live_list = df_live['Address'].fillna('').tolist()
    staging_list = df_staging['Address'].fillna('').tolist()

    matches = match_using_model(model, live_list, staging_list)

    results = pd.DataFrame(matches, columns=['From', 'To', 'Similarity'])

    results_file = os.path.join(app.config['UPLOAD_FOLDER'], 'results.xlsx')
    results.to_excel(results_file, index=False)
    matches_list = results.values.tolist()
    return render_template('results.html', matches=matches_list)

@app.route('/download')
def download():
    path = os.path.join(app.config['UPLOAD_FOLDER'], 'results.xlsx')
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
