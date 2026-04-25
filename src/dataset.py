import difflib
from pathlib import Path

import pandas as pd

# src/ is one level below the project root, so go up two levels to reach data/
_DATA_PATH = Path(__file__).parent.parent / "data" / "spotify_tracks.csv"

try:
    _raw = pd.read_csv(_DATA_PATH)
    # Two dedup passes, both keeping the highest-popularity row:
    # 1. track_id       — same recording duplicated across genres/playlists
    # 2. (track_name, artists) — same song with different album entries and IDs
    _df = (
        _raw.sort_values("popularity", ascending=False)
            .drop_duplicates(subset=["track_id"])
            .drop_duplicates(subset=["track_name", "artists"])
            .reset_index(drop=True)
    )
    _genres: list[str] = _df["track_genre"].unique().tolist()
except FileNotFoundError:
    raise FileNotFoundError(
        f"Dataset not found at {_DATA_PATH}. "
        "Please download the Spotify Tracks Dataset from Kaggle "
        "(https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) "
        "and place it at data/spotify_tracks.csv"
    )


def get_song_features(song_name: str) -> pd.Series | None:
    """
    Search the dataset for a song by name (case-insensitive exact match).
    If multiple rows match, return the one with the highest popularity score.
    Returns None if no match is found.
    """
    mask = _df["track_name"].str.lower() == song_name.lower()
    matches = _df[mask]
    if matches.empty:
        return None
    return matches.loc[matches["popularity"].idxmax()]


def match_genre(llm_genre: str) -> str | None:
    """
    Find the closest genre label from the dataset using difflib.
    Uses cutoff=0.6.  Returns the best-match string, or None if nothing
    clears the threshold.
    """
    results = difflib.get_close_matches(llm_genre, _genres, n=1, cutoff=0.6)
    return results[0] if results else None


def get_dataframe() -> pd.DataFrame:
    """Return the full dataset dataframe (read-only intended)."""
    return _df


def get_genres() -> list[str]:
    """Return the list of unique track_genre values found in the dataset."""
    return _genres
