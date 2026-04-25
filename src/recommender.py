from dataclasses import dataclass
from typing import List, Dict

import pandas as pd


@dataclass
class UserProfile:
    """Represents a user's music taste preferences."""
    favorite_genre: str              # matched against dataset genre labels
    target_energy: float             # 0.0 – 1.0
    target_tempo_bpm: float          # raw BPM 0–250
    target_valence: float            # 0.0 – 1.0
    target_danceability: float       # 0.0 – 1.0
    target_acousticness: float       # 0.0 – 1.0
    target_speechiness: float        # 0.0 – 1.0
    target_instrumentalness: float   # 0.0 – 1.0
    target_liveness: float           # 0.0 – 1.0
    target_loudness: float           # -60 to 0 dB
    target_mode: int                 # 0 = minor, 1 = major
    feature_weights: dict            # e.g. {"genre": 0.8, "energy": 0.5, ...}


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with all scoring features normalised to [0, 1].

    Already-normalised features (0-1 range): energy, valence, danceability,
    acousticness, speechiness, instrumentalness, liveness, mode.
    tempo:    divide by 250
    loudness: (loudness + 60) / 60
    """
    out = df.copy()
    out["energy_n"]           = out["energy"]
    out["valence_n"]          = out["valence"]
    out["danceability_n"]     = out["danceability"]
    out["acousticness_n"]     = out["acousticness"]
    out["speechiness_n"]      = out["speechiness"]
    out["instrumentalness_n"] = out["instrumentalness"]
    out["liveness_n"]         = out["liveness"]
    out["mode_n"]             = out["mode"]
    out["tempo_n"]            = out["tempo"] / 250.0
    out["loudness_n"]         = (out["loudness"] + 60.0) / 60.0
    return out


def _normalize_target(profile: UserProfile) -> dict:
    """Return a dict of normalised target values for the UserProfile."""
    return {
        "energy_n":           profile.target_energy,
        "valence_n":          profile.target_valence,
        "danceability_n":     profile.target_danceability,
        "acousticness_n":     profile.target_acousticness,
        "speechiness_n":      profile.target_speechiness,
        "instrumentalness_n": profile.target_instrumentalness,
        "liveness_n":         profile.target_liveness,
        "mode_n":             float(profile.target_mode),
        "tempo_n":            profile.target_tempo_bpm / 250.0,
        "loudness_n":         (profile.target_loudness + 60.0) / 60.0,
    }


def recommend(profile: UserProfile, df: pd.DataFrame, k: int = 5) -> List[Dict]:
    """
    Score every song in df against profile using weighted feature matching.

    For each numeric feature:
        contribution = weight * (1 - abs(song_normalised - target_normalised))

    Genre is a binary match:
        contribution = weight * (1 if genre matches else 0)

    Returns the top-k songs as a list of dicts:
        {track_name, artists, score}
    with scores normalised to 0–100 %.
    """
    weights = profile.feature_weights
    norm_df = _normalize(df)
    targets = _normalize_target(profile)

    # Build score column via vectorised operations
    scores = pd.Series(0.0, index=norm_df.index)

    feature_col_map = {
        "energy":           "energy_n",
        "tempo":            "tempo_n",
        "valence":          "valence_n",
        "danceability":     "danceability_n",
        "acousticness":     "acousticness_n",
        "speechiness":      "speechiness_n",
        "instrumentalness": "instrumentalness_n",
        "liveness":         "liveness_n",
        "loudness":         "loudness_n",
        "mode":             "mode_n",
    }

    for feature, col in feature_col_map.items():
        w = weights.get(feature, 0.0)
        if w == 0.0:
            continue
        diff = (norm_df[col] - targets[col]).abs()
        scores += w * (1.0 - diff)

    # Genre binary match.
    # profile.favorite_genre is guaranteed to already be a canonical dataset
    # genre label — profile_builder.py runs match_genre() before constructing
    # the UserProfile, so an exact comparison is correct here.
    genre_w = weights.get("genre", 0.0)
    if genre_w > 0.0:
        genre_match = (
            norm_df["track_genre"].str.lower() == profile.favorite_genre.lower()
        ).astype(float)
        scores += genre_w * genre_match

    # Relative normalisation to 0–100 %.
    # Dividing by the highest raw score means the best-matching song always
    # displays as 100 % and every other song is expressed relative to it.
    # (Alternative: divide by sum-of-weights for an absolute quality score,
    # but that can produce uniformly low numbers when weights are small.)
    # Guard against the degenerate case where all weights are 0.
    max_score = scores.max()
    if max_score > 0:
        pct_scores = (scores / max_score) * 100.0
    else:
        pct_scores = scores * 0.0

    # Pull a larger candidate pool then enforce one song per artist.
    # Artists field may contain collaborators separated by ";" — split and
    # check each name individually so collaborations don't sneak past the filter.
    candidate_idx = pct_scores.nlargest(min(k * 20, len(pct_scores))).index
    seen_artists: set[str] = set()
    top_idx = []
    for idx in candidate_idx:
        artists = [a.strip() for a in str(norm_df.loc[idx, "artists"]).split(";")]
        if not any(a in seen_artists for a in artists):
            seen_artists.update(artists)
            top_idx.append(idx)
        if len(top_idx) == k:
            break

    results = []
    for idx in top_idx:
        row = norm_df.loc[idx]
        results.append({
            "track_name":       row["track_name"],
            "artists":          row["artists"],
            "track_genre":      row["track_genre"],
            "score":            round(float(pct_scores[idx]), 2),
            "energy":           round(float(row["energy"]), 4),
            "tempo":            round(float(row["tempo"]), 1),
            "valence":          round(float(row["valence"]), 4),
            "danceability":     round(float(row["danceability"]), 4),
            "acousticness":     round(float(row["acousticness"]), 4),
            "speechiness":      round(float(row["speechiness"]), 4),
            "instrumentalness": round(float(row["instrumentalness"]), 4),
            "liveness":         round(float(row["liveness"]), 4),
            "loudness":         round(float(row["loudness"]), 2),
            "mode":             int(row["mode"]),
        })

    return results
