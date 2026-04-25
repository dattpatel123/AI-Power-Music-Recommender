"""
profile_builder.py — Agent Step 1.

Accepts a single natural-language string.  A lightweight LLM extraction step
pulls out any song/artist names the user mentioned, then the dataset is searched
for real audio features.  Based on what is found, one of three scenarios runs:

  1. No songs found in dataset  → LLM infers profile purely from text description
  2. Songs found, text is just names (no descriptive content) → songs-only profile
  3. Songs found + descriptive text  → blended profile (ratio by song count)
"""
import json
import os
import statistics
from pathlib import Path

import google.genai as genai
from dotenv import load_dotenv

from .dataset import get_song_features, match_genre
from .guardrails import validate_and_build
from .logger import log_event
from .recommender import UserProfile

load_dotenv(Path(__file__).parent.parent / ".env")

try:
    _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    _MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
except KeyError:
    raise EnvironmentError("GEMINI_API_KEY not found. Add it to your .env file.")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FEATURE_DEFS = """
Spotify audio feature definitions:
- energy: Perceptual intensity and activity, 0.0–1.0. High = fast/loud/noisy. Low = calm/soft.
- tempo: Estimated BPM, 0–250. Slow = 60–80, moderate = 80–120, fast = 120–180+.
- valence: Musical positiveness, 0.0–1.0. High = happy/euphoric. Low = sad/angry.
- danceability: Suitability for dancing, 0.0–1.0. Based on tempo, rhythm, beat strength.
- acousticness: Confidence the track is acoustic, 0.0–1.0. 1.0 = highly acoustic.
- speechiness: Presence of spoken words. >0.66 = mostly speech. 0.33–0.66 = mix. <0.33 = music.
- instrumentalness: Likelihood of no vocals, 0.0–1.0. >0.5 = likely instrumental.
- liveness: Presence of live audience, 0.0–1.0. >0.8 = likely live performance.
- loudness: Overall loudness in dB, range -60 to 0. Closer to 0 = louder.
- mode: 1 = major (bright/happy). 0 = minor (dark/moody).
""".strip()

_JSON_SCHEMA = """{
  "favorite_genre": "string",
  "target_energy": 0.0,
  "target_tempo_bpm": 0.0,
  "target_valence": 0.0,
  "target_danceability": 0.0,
  "target_acousticness": 0.0,
  "target_speechiness": 0.0,
  "target_instrumentalness": 0.0,
  "target_liveness": 0.0,
  "target_loudness": 0.0,
  "target_mode": 0,
  "feature_weights": {
    "genre": 0.0,
    "energy": 0.0,
    "tempo": 0.0,
    "valence": 0.0,
    "danceability": 0.0,
    "acousticness": 0.0,
    "speechiness": 0.0,
    "instrumentalness": 0.0,
    "liveness": 0.0,
    "loudness": 0.0,
    "mode": 0.0
  }
}"""

_NUMERIC_FEATURES = [
    "energy", "tempo", "valence", "danceability", "acousticness",
    "speechiness", "instrumentalness", "liveness", "loudness", "mode",
]


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict | list:
    """Strip markdown fences if present, then parse JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1:]
        text = text[:text.rfind("```")].strip()
    return json.loads(text)


def _call_llm(prompt: str) -> dict:
    try:
        response = _client.models.generate_content(model=_MODEL, contents=prompt)
        return _parse_json(response.text)
    except json.JSONDecodeError as e:
        log_event("error", {"reason": "json_parse_failed", "detail": str(e)})
        raise RuntimeError(f"LLM returned invalid JSON: {e}")
    except Exception as e:
        log_event("error", {"reason": "llm_call_failed", "detail": str(e)})
        raise RuntimeError(f"LLM call failed: {e}")


# ---------------------------------------------------------------------------
# Step 1 — entity extraction
# ---------------------------------------------------------------------------

def _extract_mentions(user_text: str) -> tuple[list[str], bool]:
    """
    Ask the LLM to pull out any song/artist names from the user's text and
    decide whether the text contains descriptive content beyond just names.

    Returns:
        mentions:        list of song/artist name strings to search
        has_description: True if the text contains mood/vibe/preference language
    """
    prompt = f"""You are a music text analyzer. Read the user's input and:
1. Extract any song titles or artist names mentioned (exact names as written).
2. Decide if the text contains descriptive content — mood, feelings, activity, or music preferences — beyond just naming songs or artists.

Return ONLY valid JSON in this exact format, no markdown, no explanation:
{{"mentions": ["name1", "name2"], "has_description": true}}

User input: "{user_text}"
"""
    try:
        response = _client.models.generate_content(model=_MODEL, contents=prompt)
        result = _parse_json(response.text)
        return result.get("mentions", []), bool(result.get("has_description", True))
    except Exception as e:
        log_event("error", {"reason": "extraction_failed", "detail": str(e)})
        # Safe fallback: treat everything as descriptive text
        return [], True


# ---------------------------------------------------------------------------
# Song feature helpers
# ---------------------------------------------------------------------------

def _average_features(songs: list) -> dict:
    return {f: sum(float(s[f]) for s in songs) / len(songs) for f in _NUMERIC_FEATURES}


def _std_features(songs: list) -> dict:
    return {
        f: statistics.stdev([float(s[f]) for s in songs]) if len(songs) > 1 else 0.0
        for f in _NUMERIC_FEATURES
    }


def _blend_ratio(n_songs: int) -> tuple[int, int]:
    """Return (text_pct, song_pct)."""
    if n_songs >= 5:
        return 20, 80
    if n_songs >= 3:
        return 40, 60
    return 60, 40


def _genre_summary(songs: list) -> str:
    """
    Return a comma-separated string of genres from the found songs, with counts
    when a genre appears more than once.  E.g. "pop (x3), indie pop (x1)".
    """
    from collections import Counter
    counts = Counter(str(s["track_genre"]) for s in songs)
    parts = [f"{g} (x{n})" if n > 1 else g for g, n in counts.most_common()]
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _prompt_text_only(user_text: str) -> str:
    return f"""You are a music preference analyzer.

{_FEATURE_DEFS}

Based on the user's description, infer their music preferences and output a JSON profile.
Set feature_weights to reflect how important each feature seems from the description.
If only an artist or song is mentioned with no other detail, infer weights based on what audio features are most distinctive of that artist's sound.

User input: "{user_text}"

Return ONLY valid JSON matching this exact schema — no markdown, no explanation:
{_JSON_SCHEMA}"""


def _prompt_songs_only(user_text: str, avg: dict, stds: dict, genre_summary: str) -> str:
    avg_lines  = "\n".join(f"  {f}: {avg[f]:.4f}"  for f in _NUMERIC_FEATURES)
    std_lines  = "\n".join(f"  {f}: {stds[f]:.4f}" for f in _NUMERIC_FEATURES)
    return f"""You are a music preference analyzer.

{_FEATURE_DEFS}

The user's example songs have these averaged audio features:
{avg_lines}

Standard deviation across songs (low = consistent preference, high = varied):
{std_lines}

Genres of the user's example songs: {genre_summary}
For favorite_genre: if the user's input explicitly mentions a genre they prefer, use that.
Otherwise choose from the genres listed above, preferring the most frequent one.

User input: "{user_text}"

Use the averaged values as targets. Set feature_weights based on consistency:
low std → high weight (the user reliably wants this), high std → low weight.

Return ONLY valid JSON matching this exact schema — no markdown, no explanation:
{_JSON_SCHEMA}"""


def _prompt_blend(user_text: str, avg: dict, stds: dict, genre_summary: str, text_pct: int, song_pct: int) -> str:
    avg_lines = "\n".join(f"  {f}: {avg[f]:.4f}"  for f in _NUMERIC_FEATURES)
    std_lines = "\n".join(f"  {f}: {stds[f]:.4f}" for f in _NUMERIC_FEATURES)
    return f"""You are a music preference analyzer.

{_FEATURE_DEFS}

The user's input contains both a description and song references.
Blend the two sources at this ratio: {text_pct}% from the description, {song_pct}% from the song features.

User description: "{user_text}"

Averaged song features from the dataset:
{avg_lines}

Standard deviation across songs (low = consistent preference, high = varied):
{std_lines}

Genres of the user's example songs: {genre_summary}
For favorite_genre: if the user's description explicitly mentions a genre they prefer, use that.
Otherwise choose from the genres of their example songs listed above, preferring the most frequent one.

Produce one unified profile reflecting both sources at the stated ratio.
When setting feature_weights, combine the signal from the text description with the
consistency signal from the songs: a feature that is both mentioned in the text and
consistent across songs (low std) should receive a high weight.

Return ONLY valid JSON matching this exact schema — no markdown, no explanation:
{_JSON_SCHEMA}"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_profile(user_text: str) -> UserProfile:
    """
    Build a validated UserProfile from a natural-language string.

    The function handles song/artist extraction internally — callers pass
    only the raw user text (the app may concatenate the two UI fields before
    calling this).

    Args:
        user_text: Free-form natural language input from the user.

    Returns:
        A fully validated UserProfile.
    """
    # Step 1 — extract song/artist mentions and detect descriptive content
    mentions, has_description = _extract_mentions(user_text)

    # Step 2 — dataset lookup for each mention
    found_songs, not_found = [], []
    for name in mentions:
        row = get_song_features(name)
        if row is not None:
            found_songs.append(row)
        else:
            not_found.append(name)

    

    # Step 3 — choose scenario and build prompt
    if not found_songs:
        # Scenario 1: no songs found in dataset — infer entirely from text
        prompt = _prompt_text_only(user_text)

    elif not has_description:
        # Scenario 2: text is just song/artist names, no descriptive content
        avg  = _average_features(found_songs)
        stds = _std_features(found_songs)
        genre_summary = _genre_summary(found_songs)
        prompt = _prompt_songs_only(user_text, avg, stds, genre_summary)

    else:
        # Scenario 3: descriptive text + found songs — blend
        avg  = _average_features(found_songs)
        stds = _std_features(found_songs)
        genre_summary = _genre_summary(found_songs)
        text_pct, song_pct = _blend_ratio(len(found_songs))
        prompt = _prompt_blend(user_text, avg, stds, genre_summary, text_pct, song_pct)

    # Step 4 — call LLM
    raw = _call_llm(prompt)

    # Step 5 — canonicalize genre to a dataset label
    genre = raw.get("favorite_genre", "")
    matched = match_genre(genre)
    if matched:
        raw["favorite_genre"] = matched
    else:
        log_event("genre_no_match", {"llm_genre": genre})
        if "feature_weights" in raw:
            raw["feature_weights"]["genre"] = 0.0

    return validate_and_build(raw)


if __name__ == "__main__":
    # Quick test
    test_input = "I love upbeat pop songs like 'Happy' by Pharrell Williams, but I also enjoy some chill indie vibes."
    profile = build_profile(test_input)
    print(profile)