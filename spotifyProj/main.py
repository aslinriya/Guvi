import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("C:/Users/WELCOME/Desktop/pandasProject/my_env/Scripts/spotifyProj/SpotifyFeatures.csv")
print("Dataset shape:", df.shape)
print(df.head())

# ---------- Genre Classification ----------
# Encode target
le = LabelEncoder()
df['genre_encoded'] = le.fit_transform(df['genre'])

# Drop non-numeric + target columns
X = df.drop(['genre', 'genre_encoded', 'artist_name', 'track_name', 'track_id'], axis=1)

# Convert categorical columns ('key','mode','time_signature') to numeric
X = pd.get_dummies(X, columns=['key', 'mode', 'time_signature'], drop_first=True)

y_genre = df['genre_encoded']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y_genre, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("/n🎵 Genre Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ---------- Popularity Prediction (Regression) ----------
y_pop = df['popularity']
X_pop = df.drop(['genre', 'genre_encoded', 'artist_name', 'track_name', 'track_id', 'popularity'], axis=1)

X_pop = pd.get_dummies(X_pop, columns=['key', 'mode', 'time_signature'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X_pop, y_pop, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print("\n📊 Popularity Prediction Results:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
