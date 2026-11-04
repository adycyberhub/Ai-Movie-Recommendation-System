import pandas as pd
from pathlib import Path



# PROJECT PATHS


BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)



# LOAD DATASETS


def load_datasets():
    movies = pd.read_csv(RAW_DIR / "movies.csv")
    ratings = pd.read_csv(RAW_DIR / "ratings.csv")
    tags = pd.read_csv(RAW_DIR / "tags.csv")
    links = pd.read_csv(RAW_DIR / "links.csv")

    print("Datasets loaded successfully.")

    return movies, ratings, tags, links



# CLEAN MOVIES


def clean_movies(movies):
    movies = movies.copy()

    movies.drop_duplicates(inplace=True)
    movies.dropna(subset=["movieId", "title"], inplace=True)

    movies["title"] = movies["title"].astype(str).str.strip()

    movies["genres"] = (
        movies["genres"]
        .fillna("(no genres listed)")
        .astype(str)
        .str.strip()
    )

    movies.drop_duplicates(
        subset=["movieId"],
        keep="first",
        inplace=True
    )

    return movies



# CLEAN RATINGS


def clean_ratings(ratings, movies):
    ratings = ratings.copy()

    ratings.drop_duplicates(inplace=True)

    ratings.dropna(
        subset=["userId", "movieId", "rating", "timestamp"],
        inplace=True
    )

    ratings = ratings[
        ratings["rating"].between(0.5, 5.0)
    ]

    valid_movie_ids = set(movies["movieId"])

    ratings = ratings[
        ratings["movieId"].isin(valid_movie_ids)
    ]

    ratings["timestamp_datetime"] = pd.to_datetime(
        ratings["timestamp"],
        unit="s",
        errors="coerce",
        utc=True
    )

    ratings.dropna(
        subset=["timestamp_datetime"],
        inplace=True
    )

    return ratings


# CLEAN TAGS

def clean_tags(tags, movies):
    tags = tags.copy()

    tags.drop_duplicates(inplace=True)

    tags.dropna(
        subset=["userId", "movieId", "tag", "timestamp"],
        inplace=True
    )

    tags["tag"] = tags["tag"].astype(str).str.strip()

    tags = tags[tags["tag"] != ""]

    valid_movie_ids = set(movies["movieId"])

    tags = tags[
        tags["movieId"].isin(valid_movie_ids)
    ]

    return tags



# CLEAN LINKS


def clean_links(links, movies):
    links = links.copy()

    links.drop_duplicates(inplace=True)

    valid_movie_ids = set(movies["movieId"])

    links = links[
        links["movieId"].isin(valid_movie_ids)
    ]

    links["tmdbId"] = pd.to_numeric(
        links["tmdbId"],
        errors="coerce"
    ).astype("Int64")

    links["imdbId"] = pd.to_numeric(
        links["imdbId"],
        errors="coerce"
    ).astype("Int64")

    return links



# CREATE CONTENT FEATURES


def create_movie_features(movies):
    movie_features = movies.copy()

    movie_features["genres"] = (
        movie_features["genres"]
        .str.lower()
        .str.replace("|", " ", regex=False)
    )

    movie_features["content_features"] = (
        movie_features["title"].str.lower()
        + " "
        + movie_features["genres"]
    )

    return movie_features



# SAVE PROCESSED DATASETS


def save_datasets(
    movies,
    ratings,
    tags,
    links,
    movie_features
):
    movies.to_csv(
        PROCESSED_DIR / "movies_cleaned.csv",
        index=False
    )

    ratings.to_csv(
        PROCESSED_DIR / "ratings_cleaned.csv",
        index=False
    )

    tags.to_csv(
        PROCESSED_DIR / "tags_cleaned.csv",
        index=False
    )

    links.to_csv(
        PROCESSED_DIR / "links_cleaned.csv",
        index=False
    )

    movie_features.to_csv(
        PROCESSED_DIR / "movie_features.csv",
        index=False
    )

    print("Processed datasets saved successfully.")


# MAIN EXECUTION

def main():
    movies, ratings, tags, links = load_datasets()

    print("\nOriginal Dataset Sizes:")
    print("Movies:", len(movies))
    print("Ratings:", len(ratings))
    print("Tags:", len(tags))
    print("Links:", len(links))

    movies_cleaned = clean_movies(movies)

    ratings_cleaned = clean_ratings(
        ratings,
        movies_cleaned
    )

    tags_cleaned = clean_tags(
        tags,
        movies_cleaned
    )

    links_cleaned = clean_links(
        links,
        movies_cleaned
    )

    movie_features = create_movie_features(
        movies_cleaned
    )

    save_datasets(
        movies_cleaned,
        ratings_cleaned,
        tags_cleaned,
        links_cleaned,
        movie_features
    )

    print("\nFinal Dataset Sizes:")
    print("Movies:", len(movies_cleaned))
    print("Ratings:", len(ratings_cleaned))
    print("Tags:", len(tags_cleaned))
    print("Links:", len(links_cleaned))

    print("\nData preprocessing completed!")


if __name__ == "__main__":
    main()