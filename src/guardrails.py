"""
guardrails.py — validate and clamp LLM-generated profile dicts.

Clamping is always silent (no exception raised).
Every clamp is logged via logger.py.
Missing required fields raise a ValueError immediately for Streamlit to display.
"""
from logger import log_event
from recommender import UserProfile

# ---------------------------------------------------------------------------
# Field definitions
# ---------------------------------------------------------------------------

# Fields clamped to [0.0, 1.0]
_UNIT_FIELDS = [
    "target_energy",
    "target_valence",
    "target_danceability",
    "target_acousticness",
    "target_speechiness",
    "target_instrumentalness",
    "target_liveness",
]

_WEIGHT_KEYS = [
    "genre", "energy", "tempo", "valence", "danceability",
    "acousticness", "speechiness", "instrumentalness", "liveness",
    "loudness", "mode",
]

_REQUIRED_FIELDS = [
    "favorite_genre",
    *_UNIT_FIELDS,
    "target_tempo_bpm",
    "target_loudness",
    "target_mode",
    "feature_weights",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _missing_fields(raw: dict) -> list[str]:
    missing = [f for f in _REQUIRED_FIELDS if f not in raw]
    if "feature_weights" in raw:
        missing += [k for k in _WEIGHT_KEYS if k not in raw["feature_weights"]]
    return missing


def _apply_clamps(raw: dict) -> dict:
    """Clamp all numeric fields in-place and log each correction."""
    out = dict(raw)

    for field in _UNIT_FIELDS:
        original = out[field]
        clamped = _clamp(float(original), 0.0, 1.0)
        if clamped != float(original):
            log_event("clamp", {"field": field, "original": original, "clamped": clamped})
        out[field] = clamped

    # tempo: 0–250
    original = out["target_tempo_bpm"]
    clamped = _clamp(float(original), 0.0, 250.0)
    if clamped != float(original):
        log_event("clamp", {"field": "target_tempo_bpm", "original": original, "clamped": clamped})
    out["target_tempo_bpm"] = clamped

    # loudness: -60 to 0
    original = out["target_loudness"]
    clamped = _clamp(float(original), -60.0, 0.0)
    if clamped != float(original):
        log_event("clamp", {"field": "target_loudness", "original": original, "clamped": clamped})
    out["target_loudness"] = clamped

    # mode: round to nearest int then clamp to {0, 1}
    original = out["target_mode"]
    rounded = int(round(float(original)))
    clamped_mode = _clamp(rounded, 0, 1)
    if clamped_mode != original:
        log_event("clamp", {"field": "target_mode", "original": original, "clamped": clamped_mode})
    out["target_mode"] = clamped_mode

    # feature_weights: each value clamped to [0, 1].
    # Genre weight has a minimum floor of 0.5 to prevent dataset bias
    # (e.g. over-represented genres dominating purely on audio features).
    weights = dict(out["feature_weights"])
    for key in _WEIGHT_KEYS:
        lo = 0.5 if key == "genre" else 0.0
        original_w = weights[key]
        clamped_w = _clamp(float(original_w), lo, 1.0)
        if clamped_w != float(original_w):
            log_event("clamp", {
                "field": f"feature_weights.{key}",
                "original": original_w,
                "clamped": clamped_w,
            })
        weights[key] = clamped_w
    out["feature_weights"] = weights

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_and_build(raw: dict) -> UserProfile:
    """
    Validate raw LLM output, clamp out-of-range values, and return a UserProfile.

    Raises:
        ValueError: If any required field is missing.
    """
    missing = _missing_fields(raw)
    if missing:
        log_event("error", {"reason": "missing_fields", "missing": missing})
        raise ValueError(
            f"LLM response is missing required fields: {missing}. "
            "Please try rephrasing your request."
        )

    clamped = _apply_clamps(raw)

    return UserProfile(
        favorite_genre=clamped["favorite_genre"],
        target_energy=clamped["target_energy"],
        target_tempo_bpm=clamped["target_tempo_bpm"],
        target_valence=clamped["target_valence"],
        target_danceability=clamped["target_danceability"],
        target_acousticness=clamped["target_acousticness"],
        target_speechiness=clamped["target_speechiness"],
        target_instrumentalness=clamped["target_instrumentalness"],
        target_liveness=clamped["target_liveness"],
        target_loudness=clamped["target_loudness"],
        target_mode=clamped["target_mode"],
        feature_weights=clamped["feature_weights"],
    )
