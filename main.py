"""Command-line entry point for the AI Movie Recommendation System."""
from pathlib import Path
import sys

from src.recommender import MovieRecommender, DEFAULT_DATA_PATH


def main():
    print("=" * 55)
    print("           AI MOVIE RECOMMENDATION SYSTEM")
    print("=" * 55)

    if not DEFAULT_DATA_PATH.exists():
        print("\nProcessed movie data was not found.")
        print("Run the preprocessing step first:")
        print("  python src/data_preprocessing.py")
        return 1

    try:
        recommender = MovieRecommender.from_csv()
    except (FileNotFoundError, ValueError) as error:
        print(f"Could not load the recommendation model: {error}")
        return 1

    print("\nFind movies similar to a movie you enjoy.")
    print("Tip: You can enter a full title or part of a title.")
    print("Type 'q' or 'quit' to exit.")

    while True:
        try:
            movie_title = input("\nEnter a movie title: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return 0

        if movie_title.casefold() in {"q", "quit", "exit"}:
            print("Thanks for using the AI Movie Recommendation System!")
            return 0

        if not movie_title:
            print("Please enter a movie title. Empty input is not accepted.")
            continue

        try:
            recommendations = recommender.recommend(movie_title, top_n=5)
            print("\nTop 5 similar movies:")
            for number, row in enumerate(recommendations.itertuples(index=False), start=1):
                print(f"{number}. {row.title} (similarity: {row.similarity_score:.3f})")
        except ValueError as error:
            print(error)
            print("Try another title or a more specific search.")


if __name__ == "__main__":
    sys.exit(main())
