---
name: Project Context — Agentic Music Recommender
description: Full context on the music recommender upgrade project — phases, stack, file structure, key decisions
type: project
---

Upgrading an existing music recommender (src/recommender.py + src/main.py + data/songs.csv) into a full agentic AI system with Gemini LLM, Streamlit UI, and the Maharshi Pandya Spotify Tracks Dataset (~114k tracks).

**Why:** User wants a full agentic pipeline: profile builder (LLM Step 1) → matching engine → explainer (LLM Step 2), with logging and guardrails.

**Target file layout (flat, at project root):**
- app.py, recommender.py, profile_builder.py, explainer.py, dataset.py, logger.py, guardrails.py
- data/spotify_tracks.csv (user must download from Kaggle)
- logs/sessions.json
- .env (GEMINI_API_KEY)
- tests/test_recommender.py (updated in Phase 7)

**Stack:** Google Gemini gemini-2.0-flash via google-generativeai, pandas vectorized ops, Streamlit, difflib for genre matching.

**Key decisions:**
- Old src/ files preserved but new root-level files take over
- UserProfile has 12 fields (no favorite_mood), including feature_weights dict
- Scoring: normalized weighted deltas + genre binary; vectorized pandas
- 7 build phases; stop after each for user confirmation

**Phase status:**
- Phase 1: dataset.py + UserProfile — IN PROGRESS
- Phases 2–7: pending
