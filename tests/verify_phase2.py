"""
Phase 2 verification — recommender.py
Run from the project root:  pytest tests/verify_phase2.py -v
"""
import sys
from pathlib import Path

# Make src/ importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import get_dataframe
from src.recommender import UserProfile, recommend

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

_PROFILE = UserProfile(
    favorite_genre="pop",
    target_energy=0.85,
    target_tempo_bpm=120.0,
    target_valence=0.75,
    target_danceability=0.80,
    target_acousticness=0.15,
    target_speechiness=0.10,
    target_instrumentalness=0.05,
    target_liveness=0.15,
    target_loudness=-6.0,
    target_mode=0,
    feature_weights={
        "genre":           0.8,
        "energy":          0.7,
        "tempo":           0.5,
        "valence":         0.6,
        "danceability":    0.6,
        "acousticness":    0.4,
        "speechiness":     0.3,
        "instrumentalness": 0.3,
        "liveness":        0.2,
        "loudness":        0.3,
        "mode":            0.4,
    },
)

_DF = get_dataframe()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


results = recommend(_PROFILE, _DF, k=5)
# PRint results 
print("Recommended tracks:", results)

