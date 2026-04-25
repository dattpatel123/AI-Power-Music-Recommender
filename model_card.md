# Model Card: Agentic Music Recommender

---

## Limitations and Biases


A limitation is that deduplication forces each song into a single genre label (whichever genre its most-streamed album entry was tagged with on Spotify). A song that legitimately spans multiple genres — say, a track that is both "soul" and "r-n-b" — gets assigned one, which means users searching for one of those genres may miss it entirely.

The feature weights and the features are LLM-inferred. If the LLM has biases about what "energetic" or "chill" music sounds like culturally, those assumptions get baked into the profile without the user knowing. This is somewhat resolved by passing in the feature definitions but of course LLM can always hallucenate the wrong values.

Finally, the system does not consider lyrics, language, or cultural context at all. A user asking for "upbeat summer music" might get results in a language they do not understand, with no way for the system to know that matters to them.

---

## Could This Be Misused?

A user might overtrust the system's explanations as objective truth, rather than understanding that the algorithm and LLM isn't perfect. It may get things incorrect. They may assume the system deeply understands their taste since the explanations are fluent and confident, which can make them feel more authoritative than they are.

To mitigate these risks: the scoring weights and genre matching logic are fully transparent and auditable in the codebase, explanations should be labeled as AI-generated interpretations rather than facts, and the system should never be presented as a replacement for genuine music discovery.

---

## What Surprised Me About Reliability

The LLM was not making that many errors. Other than maybe API rate limits, the outputs were pretty consistent format wise. More thorough testing may need to be done, but passing in JSON schema and feature definitions enforces strict response formats. Moreover, all errors and successes are logged for auditing. 

The second surprise was dataset bias. I did not expect the genre skew to be strong enough to visibly dominate results until I tested with neutral English-language prompts like "something upbeat for a workout" and got back mostly Indian pop songs. The algorithm was working exactly as designed — the data was the problem. The fix was to emphasize genre some more so that audio features of the many Indian songs don't overdominate the results 

---

## Collaboration with AI

This entire project was built in close collaboration with Claude Code, which acted as both architect and implementer across all phases — from designing the data pipeline to writing prompts, guardrails, and the Streamlit UI.

**A helpful suggestion:** When I asked how to handle the fact that the same song appears in multiple genres in the dataset, Claude suggested a deduplication strategy which helped me remove duplicate songs from the music recommendations. 

**A flawed suggestion:** One flawed suggestion was how importing worked. It gave different suggestions throughout the project on how to deal with importing packages. Eventually it helped find a consistent way to import packages by initializing a init file and doing relative imports, which worked with both streamlit and CLI runs as well. 
