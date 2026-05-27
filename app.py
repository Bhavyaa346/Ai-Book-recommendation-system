import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import os
import random
import ast

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="IBOOKS | AI Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# PREMIUM GRID CSS
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    :root {
        --primary: #818cf8;
        --secondary: #c084fc;
        --accent: #60a5fa;
        --text: #ffffff;
        --glass: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(255, 255, 255, 0.12);
        --bg: #0f1218;
    }
    #MainMenu, footer, header {visibility: hidden;}
    .block-container { padding-top: 3rem !important; max-width: 1400px; }
    .stApp {
        background-color: #05070a;
        background-image: 
            radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.12) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(168, 85, 247, 0.12) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(59, 130, 246, 0.08) 0px, transparent 50%);
        background-attachment: fixed;
        font-family: 'Outfit', sans-serif;
        color: var(--text);
    }
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background: url('https://grainy-gradients.vercel.app/noise.svg');
        opacity: 0.03;
        pointer-events: none;
        z-index: 1;
    }
    
    h1 {
        font-size: 5.5rem !important;
        line-height: 1 !important;
        background: linear-gradient(to bottom, #ffffff 0%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        text-align: center;
        letter-spacing: -3px;
        margin-top: 4rem !important;
        margin-bottom: 0.5rem !important;
    }
    .tagline { 
        text-align: center; 
        color: #64748b; 
        font-size: 1.1rem; 
        margin-bottom: 5rem; 
        font-weight: 500; 
        letter-spacing: 5px;
        text-transform: uppercase;
        opacity: 0.6;
    }
    .stSelectbox { margin-bottom: 2rem; }
    .stSelectbox div[data-baseweb="select"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 20px !important;
        backdrop-filter: blur(10px);
    }
    
    .book-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(40px) saturate(200%);
        -webkit-backdrop-filter: blur(40px) saturate(200%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 32px;
        padding: 2.5rem;
        transition: all 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        height: 100%;
    }
    .book-card:hover {
        transform: translateY(-15px) scale(1.02);
        background: rgba(255, 255, 255, 0.07);
        border-color: rgba(255, 255, 255, 0.3);
        box-shadow: 0 30px 60px rgba(0, 0, 0, 0.4);
    }
    .book-img {
        width: 100%;
        max-width: 190px;
        aspect-ratio: 2/3;
        object-fit: cover;
        border-radius: 16px;
        margin-bottom: 1.8rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.5);
    }
    .book-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #ffffff;
        line-height: 1.4;
        margin-bottom: 0.6rem;
        letter-spacing: -0.02em;
    }
    .book-author { 
        font-size: 0.9rem; 
        color: rgba(255, 255, 255, 0.5); 
        font-weight: 400;
        margin-bottom: 1.2rem;
    }
    .book-genre { 
        font-size: 0.65rem; 
        color: #ffffff; 
        text-transform: uppercase; 
        letter-spacing: 1.5px;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(168, 85, 247, 0.2));
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 5px 15px;
        border-radius: 100px;
        font-weight: 600;
    }
    
    /* Apple-style Buttons */
    .stButton>button {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        padding: 0.8rem 2.2rem !important;
        border-radius: 18px !important;
        font-weight: 500 !important;
        color: white !important;
        font-size: 0.95rem !important;
        transition: all 0.4s cubic-bezier(0.23, 1, 0.32, 1) !important;
        letter-spacing: 0.3px;
    }
    .stButton>button:hover {
        background: rgba(255, 255, 255, 0.15) !important;
        border-color: rgba(255, 255, 255, 0.25) !important;
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.3) !important;
    }
    
    .genre-btn > button {
        background: rgba(255, 255, 255, 0.03) !important;
        color: #94a3b8 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        padding: 6px 18px !important;
        font-size: 0.75rem !important;
        border-radius: 100px !important;
        font-weight: 500 !important;
    }
    .genre-btn > button:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# DATA LOADING (FLEXIBLE)
# ==========================================
@st.cache_resource
def load_all_resources():
    fps = ["Book_Details.csv", "google_books_dataset.csv", "Books.csv"]
    path = next((p for p in fps if os.path.exists(p)), None)
    if not path: return None, None, None, []
    
    try:
        try: df = pd.read_csv(path, sep=',', encoding='utf-8', low_memory=False, on_bad_lines='skip')
        except: df = pd.read_csv(path, sep=',', encoding='latin-1', low_memory=False, on_bad_lines='skip')
            
        cols = {c.lower().replace('-', '').replace('_', ''): c for c in df.columns}
        title_col = next((cols[k] for k in cols if k in ['booktitle', 'title']), df.columns[1])
        author_col = next((cols[k] for k in cols if k in ['author', 'bookauthor']), df.columns[2])
        img_col = next((cols[k] for k in cols if 'coverimage' in k or 'thumbnail' in k), None)
        desc_col = next((cols[k] for k in cols if 'summary' in k or 'description' in k), None)
        genre_col = next((cols[k] for k in cols if 'genre' in k), None)

        df = df.rename(columns={title_col: 'Book-Title', author_col: 'Book-Author'})
        
        # EXCLUSION FILTER (User Request)
        df = df[~df['Book-Title'].str.contains('Perfume', case=False, na=False)]
        
        if img_col: df = df.rename(columns={img_col: 'Image_URL'})
        else: df['Image_URL'] = ""
        
        if desc_col: df = df.rename(columns={desc_col: 'Description'})
        else: df['Description'] = ""

        if genre_col: df = df.rename(columns={genre_col: 'Genre'})
        else: df['Genre'] = "[]"

        df = df.dropna(subset=['Book-Title', 'Book-Author']).drop_duplicates(subset=['Book-Title']).head(15000)
        
        # Build Top Genres
        all_genres = []
        for g in df['Genre'].dropna():
            try: 
                parsed = ast.literal_eval(g)
                if isinstance(parsed, list): all_genres.extend(parsed)
            except: pass
        
        counts = pd.Series(all_genres).value_counts()
        top_genres = counts.head(5).index.tolist()
        
        # Manual Overrides for Diversity
        def swap_genre(target, replacement):
            nonlocal top_genres
            if target in top_genres and replacement in counts.index:
                top_genres = [g if g != target else replacement for g in top_genres]
            elif replacement in counts.index and replacement not in top_genres:
                # If target not in top 5, but replacement is valid, try to fit it in
                top_genres[-1] = replacement

        swap_genre('Young Adult', 'Horror')
        swap_genre('Fiction', 'Thriller')

        features = (df['Book-Title'].astype(str) + " " + df['Book-Author'].astype(str) + " " + df['Description'].astype(str)).fillna('')
        tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
        matrix = tfidf.fit_transform(features)
        model = NearestNeighbors(metric='cosine', algorithm='brute').fit(matrix)
        
        return df, model, matrix, top_genres
    except: return None, None, None, []

def render_book_card(row, is_hero=False):
    placeholder = "https://images.unsplash.com/photo-1543002588-bfa74002ed7e?auto=format&fit=crop&q=80&w=150&h=200"
    img = row.get('Image_URL', placeholder)
    if not isinstance(img, str) or len(img) < 5: img = placeholder
    
    style = 'style="border-color: var(--primary); padding: 2.5rem; transform: scale(1.05);"' if is_hero else ''
    img_style = 'style="max-width: 250px; height: 350px;"' if is_hero else ''
    
    # Parse Genre safely
    genre_text = ""
    g = row.get('Genre', '[]')
    try:
        parsed = ast.literal_eval(g)
        if isinstance(parsed, list) and len(parsed) > 0:
            genre_text = parsed[0]
        else:
            genre_text = str(g).strip('[]').strip("'").split(',')[0]
    except:
        genre_text = str(g).strip('[]').strip("'").split(',')[0]
    
    if genre_text == "nan" or not genre_text: genre_text = "General"

    return f"""
    <div class="book-card" {style}>
        <img src="{img}" class="book-img" {img_style} onerror="this.src='{placeholder}'">
        <div class="book-title" style="font-size: {'1.3rem' if is_hero else '1rem'};">{row['Book-Title']}</div>
        <div class="book-author" style="font-size: {'1rem' if is_hero else '0.85rem'};">{row['Book-Author']}</div>
        <div class="book-genre">{genre_text}</div>
    </div>
    """

def main():
    st.markdown('<div><h1>IBOOKS</h1><p class="tagline">Personalized AI Book Curation</p></div>', unsafe_allow_html=True)
    
    df, model, matrix, top_genres = load_all_resources()
    if df is None:
        st.error("Error: Dataset not found.")
        return

    # --- SEARCH SECTION ---
    c1, mid, c3 = st.columns([1, 4, 1])
    with mid:
        selected_book = st.selectbox("Search for a book...", [""] + list(df['Book-Title']), label_visibility="collapsed")
        
        # Action Buttons
        row_btns = st.columns([1.2, 1], gap="medium")
        with row_btns[0]: trigger = st.button("DISCOVER", use_container_width=True, key="search_btn")
        with row_btns[1]: surprise = st.button("✨ SURPRISE ME", use_container_width=True, key="surprise_btn")
        
        # Genre Buttons (Improved Layout)
        st.write("")
        selected_genre = None
        g_cols = st.columns(len(top_genres))
        for i, genre in enumerate(top_genres):
            with g_cols[i]:
                if st.button(genre.upper(), key=f"gen_{genre}", use_container_width=True):
                    selected_genre = genre

    # --- LOGIC HANDLING ---
    if surprise:
        selected_book = random.choice(df['Book-Title'].tolist())
        trigger = True

    if (trigger and selected_book) or selected_genre:
        try:
            if selected_genre:
                # FILTER BY GENRE
                st.markdown(f'<h2 style="text-align: center; margin: 3rem 0 2rem 0; color: #fff;">Trending in <span style="color: var(--primary);">{selected_genre}</span></h2>', unsafe_allow_html=True)
                genre_mask = df['Genre'].str.contains(selected_genre, na=False)
                results_df = df[genre_mask].head(10)
                
                grid = st.columns(5)
                for i, (_, row) in enumerate(results_df.iterrows()):
                    with grid[i % 5]:
                        st.markdown(render_book_card(row), unsafe_allow_html=True)
            else:
                # DISCOVERY VIEW (Split Layout)
                idx = df[df['Book-Title'] == selected_book].index[0]
                loc = df.index.get_loc(idx)
                _, indices = model.kneighbors(matrix[loc], n_neighbors=13)
                
                st.markdown(f'<h2 style="text-align: center; margin: 2rem 0; color: #fff; font-weight: 300;">Refining <span style="color: var(--primary); font-weight: 700;">{selected_book}</span></h2>', unsafe_allow_html=True)
                
                left, right = st.columns([1, 2.2], gap="large")
                with left:
                    row = df.iloc[loc]
                    st.markdown(render_book_card(row, is_hero=True), unsafe_allow_html=True)
                
                with right:
                    st.markdown('<h4 style="margin-bottom: 1.5rem; color: #fff; text-align: center;">Similar Masterpieces</h4>', unsafe_allow_html=True)
                    grid = st.columns(3)
                    for i, match_idx in enumerate(indices[0][1:10]):
                        with grid[i % 3]:
                            st.markdown(render_book_card(df.iloc[match_idx]), unsafe_allow_html=True)
        except Exception as e: st.error(f"Error: {e}")

if __name__ == '__main__': main()
