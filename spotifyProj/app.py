import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("C:/Users/WELCOME/Desktop/pandasProject/my_env/Scripts/spotifyProj/SpotifyFeatures.csv")

# Encode genre
le = LabelEncoder()
df['genre_encoded'] = le.fit_transform(df['genre'])

# Select features
numeric_cols = ['popularity', 'acousticness', 'danceability', 'duration_ms',
                'energy', 'instrumentalness', 'liveness', 'loudness',
                'speechiness', 'tempo', 'valence']
cat_cols = ['key', 'mode', 'time_signature']

X = df[numeric_cols + cat_cols]
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
y = df['genre_encoded']

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_scaled, y)

# -------- Streamlit UI --------
st.title("🎶 Spotify Audio Genre Classifier")
st.write("Enter track features to predict its genre")

# User input form
user_data = {}
for col in numeric_cols:
    user_data[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

# Dropdowns for categorical
user_data['key'] = st.selectbox("key", df['key'].unique())
user_data['mode'] = st.selectbox("mode", df['mode'].unique())
user_data['time_signature'] = st.selectbox("time_signature", df['time_signature'].unique())

if st.button("Predict Genre"):
    # Convert to DataFrame
    input_df = pd.DataFrame([user_data])

    # Encode categoricals same as training
    input_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)

    # Align columns with training set
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # Scale numerics
    input_scaled = scaler.transform(input_df)

    # Predict
    pred = clf.predict(input_scaled)
    st.success(f"Predicted Genre: {le.inverse_transform(pred)[0]}")
