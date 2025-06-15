import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def read_file(file_path):
    print(f"Attempting to read file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            print(f"File content length: {len(content)} characters")
            return content
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def compute_similarity(doc1_path, doc2_path):
    print("Starting similarity computation...")
    doc1_content = read_file(doc1_path)
    doc2_content = read_file(doc2_path)
    
    if not doc1_content or not doc2_content or not doc1_content.strip() or not doc2_content.strip():
        print("Error: One or both files are empty or contain only whitespace.")
        return None
    
    print("Creating TF-IDF vectors...")
    documents = [doc1_content, doc2_content]
    vectorizer = TfidfVectorizer(stop_words=None)
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Get the vocabulary (feature names)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get non-zero indices for each document to find present words
    doc1_indices = tfidf_matrix[0].indices
    doc2_indices = tfidf_matrix[1].indices
    
    # Find common words (intersection of indices)
    common_indices = set(doc1_indices) & set(doc2_indices)
    common_words = [feature_names[i] for i in common_indices]
    
    print("Computing cosine similarity...")
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    similarity_percentage = similarity[0][0] * 100
    print(f"Similarity computed: {similarity_percentage}%")
    return similarity_percentage, common_words

def main():
    print("Document Similarity Prediction - Starting")
    doc1_path = input("Enter path to first document (.txt): ")
    print(f"First file path entered: {doc1_path}")
    doc2_path = input("Enter path to second document (.txt): ")
    print(f"Second file path entered: {doc2_path}")
    
    if not os.path.exists(doc1_path) or not os.path.exists(doc2_path):
        print("Error: One or both files do not exist.")
        return
    
    similarity, common_words = compute_similarity(doc1_path, doc2_path)
    
    if similarity is not None:
        print(f"The documents are {similarity:.2f}% similar.")
        if common_words:
            print(f"Common words/phrases: {', '.join(common_words)}")
        else:
            print("No common words/phrases found.")
    else:
        print("Similarity calculation failed.")

if __name__ == "__main__":
    main()