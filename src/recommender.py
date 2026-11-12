"""TF-IDF and cosine-similarity movie recommender."""
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "processed" / "movie_features.csv"


class MovieRecommender:
    """Recommend movies with similar title and genre text."""

    def __init__(self, movies):
        required = {"title", "content_features"}
        missing = required.difference(movies.columns)
        if missing:
            raise ValueError(
                f"Missing required columns: {', '.join(sorted(missing))}"
            )
        if movies.empty:
            raise ValueError("The movie dataset is empty.")

        self.movies = movies.reset_index(drop=True).copy()
        self.movies["title"] = self.movies["title"].fillna("").astype(str)
        features = self.movies["content_features"].fillna("").astype(str)

        self.vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = self.vectorizer.fit_transform(features)
        self.similarity_matrix = cosine_similarity(tfidf_matrix)

    @classmethod
    def from_csv(cls, path=DEFAULT_DATA_PATH):
        """Load the processed movie feature CSV and create the recommender."""
        return cls(pd.read_csv(path))

    def _find_movie_index(self, movie_title):
        query = str(movie_title).strip().casefold()
        if not query:
            raise ValueError("Please enter a movie title.")

        titles = self.movies["title"].str.casefold()
        exact = self.movies.index[titles == query].tolist()
        if exact:
            return exact[0]

        partial = self.movies.index[titles.str.contains(
            query, regex=False, na=False
        )].tolist()

        if len(partial) == 1:
            return partial[0]
        if len(partial) > 1:
            choices = self.movies.loc[partial[:8], "title"].tolist()
            formatted = "\n".join(f"  - {title}" for title in choices)
            raise ValueError(
                f"More than one movie matches '{movie_title}'. "
                f"Please enter a more specific title:\n{formatted}"
            )
        raise ValueError(f"Movie not found: {movie_title}")

    def recommend(self, movie_title, top_n=5):
        """Return the top N similar movies, excluding the selected movie."""
        if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
            raise ValueError("top_n must be an integer of at least 1.")

        index = self._find_movie_index(movie_title)
        scores = self.similarity_matrix[index]
        ranked = scores.argsort()[::-1]
        ranked = [i for i in ranked if i != index][:top_n]

        result = self.movies.loc[ranked, ["title"]].copy()
        result["similarity_score"] = [float(scores[i]) for i in ranked]
        return result.reset_index(drop=True)


def recommend_movies(movie_title, top_n=5, data_path=DEFAULT_DATA_PATH):
    """Convenience function for one-off recommendations."""
    return MovieRecommender.from_csv(data_path).recommend(movie_title, top_n)


if __name__ == "__main__":
    model = MovieRecommender.from_csv()
    title = input("Enter a movie title (example: Toy Story): ").strip()
    try:
        print(model.recommend(title, 5).to_string(index=False))
    except ValueError as error:
        print(error)
