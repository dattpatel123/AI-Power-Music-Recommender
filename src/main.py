"""
End-to-end smoke test. Run from the project root:
    python -m src.main
"""
from dataset import get_dataframe
from profile_builder import build_profile
from recommender import recommend
from explainer import explain
import pandas as pd
TEST_INPUT = "I want something energetic and upbeat to work out to, maybe something like Blinding Lights"


def main() -> None:
    print(f"Input: {TEST_INPUT!r}\n")

    print("--- Building profile ---")
    profile = build_profile(TEST_INPUT)
    print(f"Genre:   {profile.favorite_genre}")
    print(f"Energy:  {profile.target_energy}  |  Tempo: {profile.target_tempo_bpm} BPM  |  Valence: {profile.target_valence}")
    print(f"Weights: {profile.feature_weights}\n")

    print("--- Recommending songs ---")
    df = get_dataframe()
    results = recommend(profile, df, k=5)
    for r in results:
        print(f"  {r['score']}%  {r['track_name']} — {r['artists']}")
    print()

    print("--- Generating explanations ---")
    explanations = explain(TEST_INPUT, results)
    for r, exp in zip(results, explanations):
        print(f"  {r['track_name']}: {exp}")


if __name__ == "__main__":
    main()
    
    
    
