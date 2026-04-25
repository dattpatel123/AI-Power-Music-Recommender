"""
explainer.py — Agent Step 2.

Given the user's original input and the top 5 matched songs, makes a single
Gemini call and returns exactly 5 explanation strings (one per song).
"""
import json
import os
from pathlib import Path

import google.genai as genai
from dotenv import load_dotenv

from logger import log_event

load_dotenv(Path(__file__).parent.parent / ".env")

try:
    _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    _MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
except KeyError:
    raise EnvironmentError("GEMINI_API_KEY not found. Add it to your .env file.")


def explain(user_text: str, results: list[dict]) -> list[str]:
    """
    Generate one explanation string per matched song.

    Args:
        user_text: The original natural-language input from the user.
        results:   Top-5 list of dicts with keys track_name, artists, score.

    Returns:
        List of exactly 5 explanation strings.

    Raises:
        RuntimeError: If the LLM call fails or does not return 5 strings.
    """
    def _song_block(i: int, r: dict) -> str:
        return (
            f"{i+1}. \"{r['track_name']}\" by {r['artists']} — {r['score']}% match\n"
            f"   genre={r['track_genre']}, energy={r['energy']}, tempo={r['tempo']} BPM, "
            f"valence={r['valence']}, danceability={r['danceability']}, "
            f"acousticness={r['acousticness']}, speechiness={r['speechiness']}, "
            f"instrumentalness={r['instrumentalness']}, liveness={r['liveness']}, "
            f"loudness={r['loudness']} dB, mode={'major' if r['mode'] == 1 else 'minor'}"
        )

    song_lines = "\n".join(_song_block(i, r) for i, r in enumerate(results))

    prompt = f"""You are a music recommendation assistant explaining why each song was recommended.

The user asked for: "{user_text}"

The top 5 matched songs with their audio features are:
{song_lines}

Write exactly 5 explanations — one per song, in order.
Each explanation should be 1–2 sentences. Reference the specific audio features above
that make the song a good fit for what the user asked for.
Keep the tone natural and conversational and don't be too over technical with specific exact numerical details.

Return ONLY a JSON array of exactly 5 strings, no markdown, no preamble:
["explanation 1", "explanation 2", "explanation 3", "explanation 4", "explanation 5"]"""

    try:
        response = _client.models.generate_content(model=_MODEL, contents=prompt)
        text = response.text.strip()
        if text.startswith("```"):
            text = text[text.index("\n") + 1:]
            text = text[:text.rfind("```")].strip()
        explanations = json.loads(text)
    except json.JSONDecodeError as e:
        log_event("error", {"reason": "explainer_json_parse_failed", "detail": str(e)})
        raise RuntimeError(f"Explainer returned invalid JSON: {e}")
    except Exception as e:
        log_event("error", {"reason": "explainer_call_failed", "detail": str(e)})
        raise RuntimeError(f"Explainer LLM call failed: {e}")

    if not isinstance(explanations, list) or len(explanations) != 5:
        log_event("error", {"reason": "explainer_wrong_count", "returned": len(explanations) if isinstance(explanations, list) else "non-list"})
        raise RuntimeError(f"Explainer must return exactly 5 strings, got: {explanations}")

    return [str(e) for e in explanations]
