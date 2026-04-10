from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import csv
@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float      # 0.0 – 1.0
    target_tempo_bpm: float   # raw BPM (e.g. 120)
    target_valence: float     # 0.0 – 1.0
    target_danceability: float  # 0.0 – 1.0
    target_acousticness: float  # 0.0 – 1.0

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        # TODO: Implement recommendation logic
        return self.songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        # TODO: Implement explanation logic
        return "Explanation placeholder"

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads song data from a CSV file and returns it as a list of dictionaries.
    Args:
        csv_path (str): The file path to the CSV file containing song data.
    Returns:
        List[Dict]: A list of dictionaries where each dictionary represents a song
        with its attributes. Integer fields include "id" and "tempo_bpm", while
        float fields include "energy", "valence", "danceability", and "acousticness".
    Raises:
        ValueError: If any field in the CSV cannot be converted to the expected type.
        FileNotFoundError: If the specified CSV file does not exist.
        IOError: If there is an error reading the file.
    """
    
    print(f"Loading songs from {csv_path}...")
    songs = []
    int_fields = {"id", "tempo_bpm"}
    float_fields = {"energy", "valence", "danceability", "acousticness"}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for field in int_fields:
                row[field] = int(row[field])
            for field in float_fields:
                row[field] = float(row[field])
            songs.append(dict(row))
    return songs

TEMPO_MIN = 60.0
TEMPO_MAX = 200.0

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.
    Required by recommend_songs() and src/main.py

    Scoring formula:
      (2 * genre_match) + (1.5 * mood_match) + (1 * energy_sim)
      + (1 * tempo_sim) + (1 * valence_sim) + (0.5 * dance_sim)
      + (0.5 * acoustic_sim)

    Categorical features score 1 if they match, 0 otherwise.
    Numerical features are normalized to [0, 1] then scored as
    1 - abs(user_value - song_value).
    """
    reasons = []

    # Categorical matches
    genre_match = 1.0 if song["genre"] == user_prefs["favorite_genre"] else 0.0
    mood_match  = 1.0 if song["mood"]  == user_prefs["favorite_mood"]  else 0.0

    if genre_match:
        reasons.append(f"genre matches ({song['genre']})")
    if mood_match:
        reasons.append(f"mood matches ({song['mood']})")

    # Numerical similarities (features already in [0,1] except tempo)
    energy_sim   = 1.0 - abs(user_prefs["target_energy"]       - song["energy"])
    valence_sim  = 1.0 - abs(user_prefs["target_valence"]      - song["valence"])
    dance_sim    = 1.0 - abs(user_prefs["target_danceability"]  - song["danceability"])
    acoustic_sim = 1.0 - abs(user_prefs["target_acousticness"]  - song["acousticness"])

    # Normalize tempo to [0, 1] before comparing
    user_tempo_norm = (user_prefs["target_tempo_bpm"] - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN)
    song_tempo_norm = (song["tempo_bpm"]              - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN)
    tempo_sim = 1.0 - abs(user_tempo_norm - song_tempo_norm)

    reasons.append(f"energy similarity: {energy_sim:.2f} (song {song['energy']:.2f} vs your {user_prefs['target_energy']:.2f})")
    reasons.append(f"tempo similarity: {tempo_sim:.2f} (song {song['tempo_bpm']} BPM vs your {user_prefs['target_tempo_bpm']:.0f} BPM)")
    reasons.append(f"valence similarity: {valence_sim:.2f} (song {song['valence']:.2f} vs your {user_prefs['target_valence']:.2f})")
    reasons.append(f"danceability similarity: {dance_sim:.2f} (song {song['danceability']:.2f} vs your {user_prefs['target_danceability']:.2f})")
    reasons.append(f"acousticness similarity: {acoustic_sim:.2f} (song {song['acousticness']:.2f} vs your {user_prefs['target_acousticness']:.2f})")

    score = (
        2.0 * genre_match
        + 1.5 * mood_match
        + 1.0 * energy_sim
        + 1.0 * tempo_sim
        + 1.0 * valence_sim
        + 0.5 * dance_sim
        + 0.5 * acoustic_sim
    )

    return score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Recommend songs based on user preferences.
    This function takes in user preferences, a list of songs, and an optional
    parameter `k` to determine the number of top recommendations to return. It
    scores each song based on the user's preferences, sorts them in descending
    order of their scores, and returns the top `k` songs along with their scores
    and explanations for the recommendations.
    Args:
        user_prefs (Dict): A dictionary containing the user's preferences.
        songs (List[Dict]): A list of dictionaries, where each dictionary represents
            a song with its attributes.
        k (int, optional): The number of top recommendations to return. Defaults to 5.
    Returns:
        List[Tuple[Dict, float, str]]: A list of tuples, where each tuple contains:
            - A dictionary representing a song.
            - A float representing the score of the song.
            - A string explaining the reasons for the recommendation.
    """
    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        explanation = ", ".join(reasons)
        scored.append((song, score, explanation))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
