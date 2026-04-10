"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs, score_song


PROFILES = {
    # --- Standard profiles ---
    "High-Energy Pop": {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.90,
        "target_tempo_bpm": 128,
        "target_valence": 0.85,
        "target_danceability": 0.80,
        "target_acousticness": 0.10,
    },
    "Chill Lofi": {
        "favorite_genre": "lofi",
        "favorite_mood": "chill",
        "target_energy": 0.35,
        "target_tempo_bpm": 75,
        "target_valence": 0.55,
        "target_danceability": 0.50,
        "target_acousticness": 0.80,
    },
    

    # --- Adversarial / edge-case profiles ---

    # High energy but sad/moody mood — tests whether numerical scores
    # can compensate for a mood mismatch and surface unexpected results.
    "High-Energy Moody": {
        "favorite_genre": "synthwave",
        "favorite_mood": "moody",
        "target_energy": 0.95,
        "target_tempo_bpm": 140,
        "target_valence": 0.95,   # high valence (joyful)
        "target_danceability": 0.80,
        "target_acousticness": 0.05,
    },

    # Genre that exists in the dataset but with completely opposing
    # numerical preferences — e.g. wants jazz but with rock-level energy.
    "Jazz but Loud": {
        "favorite_genre": "jazz",
        "favorite_mood": "relaxed",
        "target_energy": 0.95,    # jazz songs are typically low energy
        "target_tempo_bpm": 160,
        "target_valence": 0.80,
        "target_danceability": 0.85,
        "target_acousticness": 0.05,
    }

    
}


def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded {len(songs)} songs.\n")

    for profile_name, user_prefs in PROFILES.items():
        print(f"{'='*60}")
        print(f"Profile: {profile_name}")
        print(f"{'='*60}")
        recommendations = recommend_songs(user_prefs, songs, k=5)
        for song, score, explanation in recommendations:
            print(f"  {song['title']} by {song['artist']} — Score: {score:.2f}")
            print(f"  Why: {explanation}")
            print()


if __name__ == "__main__":
    main()
