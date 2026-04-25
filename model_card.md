# Model Card: Agentic Music Recommender

---

## Limitations and Biases


A limitation is that deduplication forces each song into a single genre label (whichever genre its most-streamed album entry was tagged with on Spotify). A song that legitimately spans multiple genres — say, a track that is both "soul" and "r-n-b" — gets assigned one, which means users searching for one of those genres may miss it entirely.

The feature weights and the features are LLM-inferred. If the LLM has biases about what "energetic" or "chill" music sounds like culturally, those assumptions get baked into the profile without the user knowing. This is somewhat resolved by passing in 

Finally, the system does not consider lyrics, language, or cultural context at all. A user asking for "upbeat summer music" might get results in a language they do not understand, with no way for the system to know that matters to them.

---

## Could This Be Misused?

The most realistic misuse risk is manipulation of recommendations for commercial benefit — for example, if an artist or label learned how the scoring works, they could craft song metadata to score highly for common user profiles regardless of actual fit. This is a known problem in real streaming platforms called "metadata gaming."

A subtler risk is over-reliance. If a user trusts the system's explanations as objective truth rather than LLM-generated plausibility, they may assume the system deeply understands their taste when it is really doing weighted math on 10 audio features. The explanations are fluent and confident, which can make them feel more authoritative than they are.

To mitigate these risks: the scoring weights and genre matching logic are fully transparent and auditable in the codebase, explanations should be labeled as AI-generated interpretations rather than facts, and the system should never be presented as a replacement for genuine music discovery or human curation.

---

## What Surprised Me About Reliability

The most surprising failure was how confidently the LLM returned slightly out-of-range values — things like an energy value of 1.03 or a tempo of 252 BPM. These were close enough to valid that they would have silently corrupted scores if guardrails had not been in place. The LLM was not making wild errors; it was making plausible-but-wrong outputs that are hard to catch without explicit validation.

The second surprise was dataset bias. I did not expect the genre skew to be strong enough to visibly dominate results until I tested with neutral English-language prompts like "something upbeat for a workout" and got back mostly Indian pop songs. The algorithm was working exactly as designed — the data was the problem. It was a clear lesson that correctness of code does not guarantee correctness of output.

---

## Collaboration with AI

This entire project was built in close collaboration with Claude Code, which acted as both architect and implementer across all phases — from designing the data pipeline to writing prompts, guardrails, and the Streamlit UI.

**A helpful suggestion:** When I asked how to handle the fact that the same song appears in multiple genres in the dataset, Claude suggested a two-pass deduplication strategy — first by `track_id` (same recording, multiple playlists), then by `(track_name, artists)` (same song, different album entries with different IDs). This was genuinely insightful: a single-pass dedup would have missed the second case, and Claude anticipated the edge case before I had tested for it.

**A flawed suggestion:** Early on, Claude set up the intra-package imports as relative imports (e.g. `from .logger import log_event`) — which is technically correct for a package. However, this broke the workflow of running files directly with `python3 src/main.py`, which is how I preferred to test during development. Claude initially presented relative imports as the clean solution without flagging the tradeoff, and it took a separate conversation to surface that `python -m src.main` is required when relative imports are used. The fix existed, but the tradeoff should have been stated upfront.
