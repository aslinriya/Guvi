import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("C:/Users/WELCOME/Desktop/pandasProject/my_env/Scripts/ImdbReommender/movies_full.csv", encoding="cp1252")

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df['Storyline'].fillna(""))

def recommend_movies(input_story, top_n=5):
    input_vec = vectorizer.transform([input_story])
    sim_scores = cosine_similarity(input_vec, X).flatten()
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    return df.iloc[top_indices]
