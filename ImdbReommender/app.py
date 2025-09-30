import streamlit as st
import pandas as pd
from recommender import recommend_movies

st.title("🎬 IMDb Movie Recommendation System")

story_input = st.text_area("Enter a movie storyline:")

if st.button("Recommend"):
    if story_input.strip():
        results = recommend_movies(story_input)
        st.subheader("Top 5 Recommended Movies:")
        for i, row in results.iterrows():
            st.write(f"**{row['Movie Name']}** - {row['Storyline']}")
    else:
        st.warning("Please enter a storyline.")
