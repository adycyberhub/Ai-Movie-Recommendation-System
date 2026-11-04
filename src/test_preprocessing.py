import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

movies = pd.read_csv(DATA_DIR / "movies_cleaned.csv")
ratings = pd.read_csv(DATA_DIR / "ratings_cleaned.csv")
tags = pd.read_csv(DATA_DIR / "tags_cleaned.csv")
links = pd.read_csv(DATA_DIR / "links_cleaned.csv")
movie_features = pd.read_csv(DATA_DIR / "movie_features.csv")

assert movies["movieId"].is_unique
assert movies["title"].notna().all()

assert ratings["rating"].between(0.5, 5.0).all()
assert ratings["movieId"].isin(movies["movieId"]).all()

assert tags["movieId"].isin(movies["movieId"]).all()
assert links["movieId"].isin(movies["movieId"]).all()

assert "content_features" in movie_features.columns

print("All preprocessing tests passed!")