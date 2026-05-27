import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import os

def load_books(filepath):
    """Loads and cleans the books dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    # Load with common settings for this Kaggle dataset
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(filepath, sep=',', encoding=enc, low_memory=False, on_bad_lines='skip')
            if len(df.columns) > 1:
                break
        except:
            continue
    else:
        raise ValueError("Could not parse CSV file. Check encoding or format.")

    # Canonicalize column names
    df.columns = [c.replace(' ', '-').title() for c in df.columns]
    col_map = {'Isbn': 'ISBN', 'Book-Title': 'Book-Title', 'Book-Author': 'Book-Author', 'Publisher': 'Publisher'}
    df.rename(columns={c: col_map[c] for c in df.columns if c in col_map}, inplace=True)
    
    # Drop rows with missing crucial info
    df.dropna(subset=['Book-Title', 'Book-Author', 'Publisher'], inplace=True)
    
    # Remove duplicates
    df.drop_duplicates(subset=['Book-Title'], inplace=True)
    
    return df

def prepare_content_model(df):
    """Prepares the TF-IDF matrix and KNN model for content-based filtering."""
    # Combine features into a single string for vectorization
    # We weight the Author and Title more by repeating them if needed, 
    # but for now, a simple concatenation is a good start.
    df['combined_features'] = (
        df['Book-Title'] + " " + 
        df['Book-Author'] + " " + 
        df['Publisher']
    ).fillna('')
    
    # Use TF-IDF to vectorize the text data
    tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    
    # Train KNN model
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(tfidf_matrix)
    
    return model, tfidf_matrix

def recommend_books(book_name, df, model, tfidf_matrix, n=5):
    """Recommends N books similar to the given book name using content similarity."""
    if book_name not in df['Book-Title'].values:
        return None
    
    # Get index of the book
    idx = df[df['Book-Title'] == book_name].index[0]
    
    # Get the vector for this book
    book_vector = tfidf_matrix[df.index.get_loc(idx)]
    
    # Find neighbors
    distances, indices = model.kneighbors(book_vector, n_neighbors=n+1)
    
    # Get matching book titles
    recommendations = []
    for i in range(1, len(indices[0])):
        recommended_idx = df.index[indices[0][i]]
        recommendations.append(df.loc[recommended_idx, 'Book-Title'])
        
    return recommendations

if __name__ == "__main__":
    # Test with a small subset or if file exists
    books_path = r'D:\Projects\Book Recommendation System\Books.csv'
    try:
        print("Loading books...")
        books_df = load_books(books_path)
        print(f"Loaded {len(books_df)} unique books.")
        
        print("Preparing model (this may take a moment)...")
        nn_model, matrix = prepare_content_model(books_df)
        
        test_book = books_df['Book-Title'].iloc[0]
        print(f"\nRecommendations for '{test_book}':")
        results = recommend_books(test_book, books_df, nn_model, matrix)
        
        if results:
            for i, r in enumerate(results, 1):
                print(f"{i}. {r}")
        else:
            print("Book not found or no recommendations.")
            
    except Exception as e:
        print(f"Error: {e}")
