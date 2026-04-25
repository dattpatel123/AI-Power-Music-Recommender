import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from src.dataset import get_dataframe
from src.explainer import explain
from src.logger import log_session
from src.profile_builder import build_profile
from src.recommender import recommend

st.set_page_config(page_title="Music Recommender", page_icon="🎵", layout="centered")
st.title("Music Recommender")
st.caption("Describe what you're in the mood for, share some songs you like, or both.")

user_text = st.text_area(
    "What are you in the mood for?",
    placeholder="e.g. Something upbeat and energetic for a workout...",
    height=100,
)

example_songs = st.text_input(
    "Example songs (optional, comma separated)",
    placeholder="e.g. Blinding Lights, Levitating, As It Was",
)

run = st.button("Find songs", type="primary")

if run:
    if not user_text.strip() and not example_songs.strip():
        st.error("Please enter a description, some example songs, or both.")
        st.stop()

    # Combine both inputs into one natural-language string for profile_builder
    combined = user_text.strip()
    if example_songs.strip():
        combined += ("\n" if combined else "") + f"Example songs I like: {example_songs.strip()}"

    try:
        with st.spinner("Building your profile..."):
            profile = build_profile(combined)

        with st.spinner("Finding matches..."):
            df = get_dataframe()
            results = recommend(profile, df, k=5)

        with st.spinner("Generating explanations..."):
            explanations = explain(combined, results)

        log_session({
            "user_text": user_text.strip(),
            "example_songs": example_songs.strip(),
            "profile": {
                "favorite_genre": profile.favorite_genre,
                "feature_weights": profile.feature_weights,
                "target_energy": profile.target_energy,
                "target_tempo_bpm": profile.target_tempo_bpm,
                "target_valence": profile.target_valence,
            },
            "results": results,
            "explanations": explanations,
        })

    except Exception as e:
        st.error(f"Something went wrong: {e}")
        st.stop()

    st.divider()
    st.subheader("Your recommendations")

    for result, explanation in zip(results, explanations):
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{result['track_name']}**")
                st.caption(result["artists"])
            with col2:
                st.metric("Match", f"{result['score']}%")
            st.write(explanation)