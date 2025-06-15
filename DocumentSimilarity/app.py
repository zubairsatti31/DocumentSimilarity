from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from docx import Document
from PyPDF2 import PdfReader
from io import BytesIO

app = Flask(__name__)

def read_file(file_content, filename):
    """Read content from uploaded file based on file type."""
    try:
        if not file_content:
            return None
        content = BytesIO(file_content)  # Create a file-like object
        if filename.endswith('.txt'):
            return content.read().decode('utf-8')
        elif filename.endswith('.docx'):
            doc = Document(content)
            return '\n'.join([para.text for para in doc.paragraphs])
        elif filename.endswith('.pdf'):
            pdf = PdfReader(content)
            text = ''
            for page in pdf.pages:
                text += page.extract_text() + '\n'
            return text
        else:
            return None
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return None

def compute_similarity(doc1_content, doc2_content):
    if not doc1_content or not doc2_content or not doc1_content.strip() or not doc2_content.strip():
        return None, []
    
    documents = [doc1_content, doc2_content]
    vectorizer = TfidfVectorizer(stop_words=None)
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    feature_names = vectorizer.get_feature_names_out()
    doc1_indices = tfidf_matrix[0].indices
    doc2_indices = tfidf_matrix[1].indices
    common_indices = set(doc1_indices) & set(doc2_indices)
    common_words = [feature_names[i] for i in common_indices]
    
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    similarity_percentage = similarity[0][0] * 100
    return similarity_percentage, common_words

@app.route('/', methods=['GET', 'POST'])
def index():
    similarity = None
    common_words = []
    error = None
    
    if request.method == 'POST':
        if 'file1' not in request.files or 'file2' not in request.files:
            error = "Please upload two files."
        else:
            file1 = request.files['file1']
            file2 = request.files['file2']
            
            if file1.filename == '' or file2.filename == '':
                error = "Please select valid files."
            else:
                try:
                    doc1_content = read_file(file1.read(), file1.filename)
                    doc2_content = read_file(file2.read(), file2.filename)
                    if doc1_content and doc2_content:
                        similarity, common_words = compute_similarity(doc1_content, doc2_content)
                    else:
                        error = "Failed to read one or both files."
                except Exception as e:
                    error = f"Error processing files: {e}"
    
    return render_template('index.html', similarity=similarity, common_words=common_words, error=error)

if __name__ == '__main__':
    app.run(debug=True)