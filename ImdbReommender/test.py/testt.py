import requests
import pandas as pd

API_KEY = "http://www.omdbapi.com/?i=tt3896198&apikey=b4684dbd"  # 🔑 replace with your OMDb API key
BASE_URL = "http://www.omdbapi.com/"

def fetch_movies_by_genre(genre, limit=50):
    """
    Fetch movies by genre using OMDb API (based on IMDb data).
    """
    movies = []
    page = 1
    while len(movies) < limit:
        params = {
            "apikey": API_KEY,
            "s": genre,      # search by keyword (genre as search term)
            "type": "movie",
            "page": page
        }
        response = requests.get(BASE_URL, params=params)
        data = response.json()

        if data.get("Response") == "False":
            break

        for item in data.get("Search", []):
            imdb_id = item.get("imdbID")

            # fetch full details
            details_params = {"apikey": API_KEY, "i": imdb_id, "plot": "short"}
            details = requests.get(BASE_URL, params=details_params).json()

            movies.append({
                "Title": details.get("Title", "N/A"),
                "Year": details.get("Year", "N/A"),
                "Rating": details.get("imdbRating", "N/A"),
                "Genre": details.get("Genre", "N/A"),
                "Plot": details.get("Plot", "N/A"),
            })

            if len(movies) >= limit:
                break
        page += 1
    return movies


# Example genres
genres = ["Action", "Comedy", "Drama", "Thriller", "Animation"]

all_movies = []
for g in genres:
    print(f"Fetching {g} movies...")
    all_movies.extend(fetch_movies_by_genre(g, limit=50))  # 50 per genre

# Save to CSV
df = pd.DataFrame(all_movies)
df.to_csv("imdb_movies_by_genre.csv", index=False)
print("✅ Saved imdb_movies_by_genre.csv")
print("URL:", response.url)
print("Response JSON:", data)
