import pandas as pd
import pytest
from recommender import MovieRecommender


@pytest.fixture
def sample_movies():
    return pd.DataFrame({
        "title": [
            "Toy Story (1995)",
            "Toy Story 2 (1999)",
            "Finding Nemo (2003)",
            "The Matrix (1999)",
        ],
        "content_features": [
            "toy story adventure animation children comedy",
            "toy story sequel adventure animation children comedy",
            "finding nemo adventure animation children",
            "matrix science fiction action",
        ],
    })


def test_returns_similar_movies(sample_movies):
    model = MovieRecommender(sample_movies)
    result = model.recommend("Toy Story (1995)", 2)
    assert len(result) == 2
    assert "Toy Story (1995)" not in result.title.tolist()
    assert result.iloc[0].title == "Toy Story 2 (1999)"
    assert result.similarity_score.between(0, 1).all()


def test_case_insensitive_title(sample_movies):
    result = MovieRecommender(sample_movies).recommend("toy story (1995)", 1)
    assert len(result) == 1


def test_unique_partial_title(sample_movies):
    result = MovieRecommender(sample_movies).recommend("Finding Nemo", 1)
    assert result.iloc[0].title != "Finding Nemo (2003)"


def test_ambiguous_partial_title(sample_movies):
    with pytest.raises(ValueError, match="more than one movie"):
        MovieRecommender(sample_movies).recommend("Toy Story")


def test_empty_title(sample_movies):
    with pytest.raises(ValueError, match="Please enter"):
        MovieRecommender(sample_movies).recommend("")


def test_unknown_title(sample_movies):
    with pytest.raises(ValueError, match="Movie not found"):
        MovieRecommender(sample_movies).recommend("Unknown Movie")


@pytest.mark.parametrize("invalid_top_n", [0, -1, 1.5, True])
def test_invalid_top_n(sample_movies, invalid_top_n):
    with pytest.raises(ValueError, match="top_n"):
        MovieRecommender(sample_movies).recommend(
            "Toy Story (1995)", invalid_top_n
        )


def test_actual_dataset():
    from pathlib import Path

    path = (
        Path(__file__).resolve().parent.parent
        / "data" / "processed" / "movie_features.csv"
    )
    model = MovieRecommender.from_csv(path)
    result = model.recommend("Toy Story (1995)", 5)
    assert len(result) == 5
    assert "Toy Story (1995)" not in result.title.tolist()
