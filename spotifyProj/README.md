# Classifying the Audio Genres

## 📌 Problem Statement
The project focuses on building machine learning models to **classify audio tracks by genre** and predict their **popularity** using Spotify audio features. The dataset contains multiple numerical and categorical features that describe the musical properties of each track.

## 🎯 Business Use Cases
1. **Recommendation System** – Suggest music to users based on their listening preferences.  
2. **Genre Classification** – Automatically classify tracks into genres.  
3. **Popularity Prediction** – Predict the popularity of a song based on its features.  
4. **Feature Importance Analysis** – Identify key features influencing popularity and genre.  
5. **Clustering Analysis** – Group similar tracks into clusters for playlist generation.  
6. **Trend Analysis** – Study how music features evolve over time.  

## 📊 Dataset Description
The dataset is derived from **Spotify’s Web API**.  
Key columns include:  
- `track_id`, `track_name`, `artists`, `album_name`  
- Audio features: `danceability`, `energy`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`  
- Metadata: `popularity`, `duration_ms`, `explicit`, `key`, `mode`, `loudness`, `time_signature`, `track_genre`  

## 🛠️ Approach
1. **Exploratory Data Analysis (EDA):** Clean and visualize the dataset.  
2. **Feature Engineering:** Handle missing values, normalize features, encode categorical data.  
3. **Modeling:** Train ML models for classification (genre) and regression (popularity).  
4. **Evaluation:**  
   - Classification → Accuracy, Precision, Recall, F1 Score  
   - Regression → MAE, MSE, R²  
5. **Deployment:** Build a **Streamlit app** to allow users to explore and interact with the models.  

## 🚀 Technology Used
- **Python**: Core programming language  
- **Pandas, NumPy**: Data manipulation  
- **Matplotlib, Seaborn, Plotly**: Visualization  
- **Scikit-learn**: Machine Learning (classification, regression, clustering)  
- **SHAP / LIME**: Feature importance analysis  
- **Streamlit**: Interactive web app  

## 📈 Expected Results
- High accuracy in **genre classification**  
- Reliable **popularity predictions**  
- Meaningful **feature importance insights**  
- Clear **clustering and trend visualizations**  
- Interactive **recommendation system**  

## 📚 References
- [Spotify Web API Documentation](https://developer.spotify.com/documentation/web-api/)  
- [Scikit-learn Documentation](https://scikit-learn.org/)  
- [Streamlit Documentation](https://docs.streamlit.io/)  

## 📬 Contact
For queries, reach out at: **your.email@example.com**
