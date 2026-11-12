# AI Movie Recommendation System

A simple Python machine-learning prototype that recommends five movies similar to a movie entered by the user. It uses the MovieLens Latest Small dataset, TF-IDF text vectorization, and cosine similarity.

## Project objective

The system demonstrates **content-based filtering**:
1. Load movie metadata from CSV files.
2. Clean movie titles, genres, ratings, tags, and links.
3. Combine each movie's title and genres into text features.
4. Convert the text features into TF-IDF vectors.
5. Compare movie vectors using cosine similarity.
6. Accept a movie title and return up to five similar movies.

This prototype uses title and genre text for recommendations. Although ratings and tags are cleaned as part of preprocessing, they are not currently used by the recommendation algorithm. It does not implement collaborative filtering, a web interface, TMDB integration, or cloud deployment.

## Technology stack

- Python
- Pandas — dataset loading and preprocessing
- Scikit-learn — TF-IDF and cosine similarity
- Pytest — automated tests
- MovieLens Latest Small — movie metadata and ratings dataset

## Repository structure

```text
Ai-Movie-Recommendation-System/
├── data/
│   ├── raw/                 # Original MovieLens CSV files
│   └── processed/           # Generated cleaned CSV files and features
├── src/
│   ├── data_preprocessing.py
│   ├── recommender.py
│   ├── test_preprocessing.py
│   └── test_recommender.py
├── main.py                  # Command-line application
├── app.py                   # Reserved / currently unused
├── requirements.txt
└── README.md
```

## Setup

Python 3.10 or newer is recommended.

Create and activate a virtual environment (optional but recommended):

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the project

First, make sure the four original MovieLens files are present in `data/raw/`:

- `movies.csv`
- `ratings.csv`
- `tags.csv`
- `links.csv`

Generate the cleaned data and movie features:

```bash
python src/data_preprocessing.py
```

Start the interactive recommendation program:

```bash
python main.py
```

Enter a movie title, for example `Toy Story`. A unique partial title is accepted. If multiple movies match, the program asks for a more specific title. Enter `q` to exit.

You can also run the recommender module directly:

```bash
python -m src.recommender
```

## Example interaction

```text
=======================================================
           AI MOVIE RECOMMENDATION SYSTEM
=======================================================

Find movies similar to a movie you enjoy.
Tip: You can enter a full title or part of a title.
Type 'q' or 'quit' to exit.

Enter a movie title: Toy Story

Top 5 similar movies:
1. ... (similarity: 0.XXX)
2. ... (similarity: 0.XXX)
3. ... (similarity: 0.XXX)
4. ... (similarity: 0.XXX)
5. ... (similarity: 0.XXX)
```

The dots above are illustrative placeholders; actual titles and scores are calculated from the included dataset.

## Run tests

From the repository root:

```bash
python -m pytest -q
```

The tests cover recommendation ordering on a small sample, case-insensitive and partial title matching, unknown titles, invalid `top_n`, and preprocessing output checks.

## How the algorithm works

**TF-IDF (Term Frequency–Inverse Document Frequency)** represents each movie's combined title and genre text as a numerical vector. Terms that help distinguish one movie from others receive more weight.

**Cosine similarity** compares two vectors by measuring the cosine of the angle between them. A score closer to 1 indicates more similar text features; a score closer to 0 indicates less overlap.

The selected movie is excluded from its own recommendation list, and the highest-scoring remaining movies are returned.

## Limitations

- Recommendations are based on title and genre text, not individual user behaviour.
- The current implementation does not use ratings or tags to calculate similarity.
- Similarity scores measure text-feature overlap; they are not predicted user ratings.
- Results depend on the movies available in the MovieLens dataset.
- This is an academic prototype, not a production recommendation service.

## Team

Project code: **AAI-005**  
Programme: **BCA (Cloud Computing), Semester V**  
Team leader: **Aditya Singh**

