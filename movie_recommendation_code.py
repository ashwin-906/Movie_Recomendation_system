"""
Movie Recommendation System
============================
Uses Content-Based Filtering with TMDB 5000 Movies Dataset.

Dataset: Download from Kaggle
  - https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
  - Files needed: tmdb_5000_movies.csv, tmdb_5000_credits.csv

Install dependencies:
  pip install pandas numpy scikit-learn nltk streamlit

Usage:
  1. Place CSV files in the same directory as this script
  2. Run: python movie_recommendation_code.py
  3. For web app: streamlit run movie_recommendation_code.py
"""

import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import pickle
import os

# ============================================================
# STEP 1: Load and Merge Datasets
# ============================================================
def load_data(movies_path='tmdb_5000_movies.csv', credits_path='tmdb_5000_credits.csv'):
    """Load and merge the TMDB movies and credits datasets."""
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    
    # Merge on title
    movies = movies.merge(credits, on='title')
    
    # Keep only relevant columns
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    
    # Drop rows with missing values
    movies.dropna(inplace=True)
    
    print(f"Loaded {len(movies)} movies successfully!")
    return movies


# ============================================================
# STEP 2: Feature Extraction Helpers
# ============================================================
def convert_json_column(obj):
    """Extract 'name' field from JSON-like string columns (genres, keywords)."""
    result = []
    for item in ast.literal_eval(obj):
        result.append(item['name'])
    return result

def get_top3_cast(obj):
    """Extract top 3 cast members."""
    result = []
    for i, item in enumerate(ast.literal_eval(obj)):
        if i >= 3:
            break
        result.append(item['name'])
    return result

def get_director(obj):
    """Extract director name from crew data."""
    for item in ast.literal_eval(obj):
        if item['job'] == 'Director':
            return [item['name']]
    return []


# ============================================================
# STEP 3: Preprocess Data
# ============================================================
def preprocess(movies):
    """Apply feature extraction and create combined tags."""
    # Convert JSON columns
    movies['genres'] = movies['genres'].apply(convert_json_column)
    movies['keywords'] = movies['keywords'].apply(convert_json_column)
    movies['cast'] = movies['cast'].apply(get_top3_cast)
    movies['crew'] = movies['crew'].apply(get_director)
    
    # Convert overview to list of words
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    
    # Remove spaces from multi-word names (e.g., "Sam Mendes" -> "SamMendes")
    for col in ['genres', 'keywords', 'cast', 'crew']:
        movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])
    
    # Combine all features into a single 'tags' column
    movies['tags'] = (
        movies['overview'] + 
        movies['genres'] + 
        movies['keywords'] + 
        movies['cast'] + 
        movies['crew']
    )
    
    # Create final dataframe
    new_df = movies[['movie_id', 'title', 'tags']].copy()
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())
    
    print("Preprocessing complete!")
    return new_df


# ============================================================
# STEP 4: Apply Stemming
# ============================================================
ps = PorterStemmer()

def stem_text(text):
    """Apply Porter Stemming to reduce words to root form."""
    return " ".join([ps.stem(word) for word in text.split()])


# ============================================================
# STEP 5: Build Recommendation Model
# ============================================================
def build_model(df):
    """Vectorize tags and compute cosine similarity matrix."""
    # Apply stemming
    df['tags'] = df['tags'].apply(stem_text)
    
    # Vectorize using CountVectorizer (top 5000 features, remove English stop words)
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df['tags']).toarray()
    
    # Compute cosine similarity
    similarity = cosine_similarity(vectors)
    
    print(f"Model built! Similarity matrix shape: {similarity.shape}")
    return similarity


# ============================================================
# STEP 6: Recommend Movies
# ============================================================
def recommend(movie_title, df, similarity, n=5):
    """
    Get top N movie recommendations based on content similarity.
    
    Args:
        movie_title: Title of the movie to find similar movies for
        df: DataFrame with movie data
        similarity: Cosine similarity matrix
        n: Number of recommendations (default: 5)
    
    Returns:
        List of recommended movie titles
    """
    # Find movie index
    matches = df[df['title'].str.lower() == movie_title.lower()]
    if matches.empty:
        print(f"Movie '{movie_title}' not found in database.")
        # Try partial match
        partial = df[df['title'].str.lower().str.contains(movie_title.lower())]
        if not partial.empty:
            print(f"Did you mean: {', '.join(partial['title'].head(5).tolist())}?")
        return []
    
    movie_index = matches.index[0]
    
    # Get similarity scores and sort
    distances = similarity[movie_index]
    movie_list = sorted(
        list(enumerate(distances)), 
        reverse=True, 
        key=lambda x: x[1]
    )[1:n+1]  # Skip first (itself)
    
    recommendations = []
    print(f"\nTop {n} recommendations for '{df.iloc[movie_index]['title']}':")
    print("-" * 50)
    for i, (idx, score) in enumerate(movie_list, 1):
        title = df.iloc[idx]['title']
        recommendations.append(title)
        print(f"  {i}. {title} (similarity: {score:.4f})")
    
    return recommendations


# ============================================================
# STEP 7: Save/Load Model
# ============================================================
def save_model(df, similarity, path='model'):
    """Save model artifacts for later use."""
    os.makedirs(path, exist_ok=True)
    pickle.dump(df, open(f'{path}/movie_dict.pkl', 'wb'))
    pickle.dump(similarity, open(f'{path}/similarity.pkl', 'wb'))
    print(f"Model saved to '{path}/' directory!")

def load_model(path='model'):
    """Load saved model artifacts."""
    df = pickle.load(open(f'{path}/movie_dict.pkl', 'rb'))
    similarity = pickle.load(open(f'{path}/similarity.pkl', 'rb'))
    print("Model loaded successfully!")
    return df, similarity


# ============================================================
# MAIN: Run the Pipeline
# ============================================================
if __name__ == '__main__':
    import sys
    
    # Check if running as Streamlit app
    if 'streamlit' in sys.modules or any('streamlit' in arg for arg in sys.argv):
        import streamlit as st
        
        st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
        st.title("🎬 Movie Recommendation System")
        st.markdown("*Content-Based Filtering using TMDB 5000 Dataset*")
        
        @st.cache_resource
        def load_cached_model():
            if os.path.exists('model/movie_dict.pkl'):
                return load_model()
            else:
                movies = load_data()
                df = preprocess(movies)
                sim = build_model(df)
                save_model(df, sim)
                return df, sim
        
        try:
            df, similarity = load_cached_model()
            selected = st.selectbox("Select a movie:", df['title'].values)
            
            if st.button("Get Recommendations", type="primary"):
                recs = recommend(selected, df, similarity)
                st.subheader("Recommended Movies:")
                for i, movie in enumerate(recs, 1):
                    st.write(f"**{i}.** {movie}")
        except FileNotFoundError:
            st.error("Dataset files not found! Place tmdb_5000_movies.csv and tmdb_5000_credits.csv in the same directory.")
    
    else:
        # Command-line mode
        print("=" * 60)
        print("  MOVIE RECOMMENDATION SYSTEM")
        print("  Content-Based Filtering with TMDB 5000 Dataset")
        print("=" * 60)
        
        try:
            # Load and process data
            movies = load_data()
            df = preprocess(movies)
            similarity = build_model(df)
            
            # Save model
            save_model(df, similarity)
            
            # Demo recommendations
            demo_movies = ['Avatar', 'The Dark Knight', 'Interstellar', 'Inception', 'Titanic']
            for movie in demo_movies:
                recommend(movie, df, similarity)
                print()
                
        except FileNotFoundError:
            print("\n[ERROR] Dataset files not found!")
            print("Please download from: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata")
            print("Place these files in the current directory:")
            print("  - tmdb_5000_movies.csv")
            print("  - tmdb_5000_credits.csv")
