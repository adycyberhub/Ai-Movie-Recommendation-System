"""TF-IDF and cosine-similarity movie recommender."""
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "processed" / "movie_features.csv"

class MovieRecommender:
    def __init__(self, movies):
        required = {"title", "content_features"}
        missing = required.difference(movies.columns)
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        self.movies = movies.reset_index(drop=True).copy()
        self.movies["title"] = self.movies["title"].astype(str)
        features = self.movies["content_features"].fillna("").astype(str)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = self.vectorizer.fit_transform(features)
        self.similarity_matrix = cosine_similarity(tfidf_matrix)

    @classmethod
    def from_csv(cls, path=DEFAULT_DATA_PATH):
        return cls(pd.read_csv(path))

    def recommend(self, movie_title, top_n=5):
        if top_n < 1:
            raise ValueError("top_n must be at least 1")
        matches = self.movies.index[
            self.movies["title"].str.casefold() == str(movie_title).casefold()
        ].tolist()
        if not matches:
            raise ValueError(f"Movie not found: {movie_title}")
        index = matches[0]
        scores = self.similarity_matrix[index]
        ranked = scores.argsort()[::-1]
        ranked = [i for i in ranked if i != index][:top_n]
        result = self.movies.loc[ranked, ["title"]].copy()
        result["similarity_score"] = [float(scores[i]) for i in ranked]
        return result.reset_index(drop=True)

def recommend_movies(movie_title, top_n=5, data_path=DEFAULT_DATA_PATH):
    return MovieRecommender.from_csv(data_path).recommend(movie_title, top_n)

if __name__ == "__main__":
    model = MovieRecommender.from_csv()
    title = input("Enter a movie title (example: Toy Story (1995)): ").strip()
    try:
        print(model.recommend(title, 5).to_string(index=False))
    except ValueError as error:
        print(error)
