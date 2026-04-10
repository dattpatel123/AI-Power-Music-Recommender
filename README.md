# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

This project is a content-based music recommender built in Python. Given a user preference profile (genre, mood, energy, tempo, valence, danceability, acousticness), it scores every song in a catalog against those preferences and returns the top matches. Categorical features (genre, mood) are compared as exact matches, while numerical features are normalized and scored by closeness. Genre is weighted most heavily (2 pts) to reflect how strongly people identify with a genre, followed by mood (1.5 pts), then energy/tempo/valence (1 pt each), and finally danceability/acousticness (0.5 pts each). The system was tested against several user profiles — including adversarial cases like mismatched energy/mood combinations and genres absent from the catalog — to expose edge cases in the scoring logic.

---

## How The System Works

Explain your design in plain language.


Some prompts to answer:

- What features does each `Song` use in your system
  - For example: genre, mood, energy, tempo
- What information does your `UserProfile` store
- How does your `Recommender` compute a score for each song
- How do you choose which songs to recommend

You can include a simple diagram or bullet list if helpful.

Real world systems use collaborative, content based, or hybrid. Collaborative uses user interaction data to find others who played songs with similar taste. Content-based uses attributes of the song itself to find other similar ones. Hybrid is a mix of the two. My system will be content-based since we are only given song data. The system will use the numerical and categorical features of the song to recommend songs. We will do a content-based filtering to help recommened songs based on the user preferences and the song metadata. UserProfile stores information about their preferences like genre, mood, energy, tempo, valence, dancebility, etc. Then the system will find a similarity score between that preference and all the songs in the playlist, giving a similarity score for all songs. Then we rank the scores from greatest to lowest, to recommend the most similar songs first. We use a points based weight system where genre match= 2points, mood match=1.5 point, energy=1 point, tempo=1 point, valence=1 point, dance=0.5, acoustic=0.5. Similarity Score = (2 * genre_match) + (1.5 * mood_match) + (1 * energy_similarity) + (1 * tempo_similarity) + (1 * valence_similarity) + (0.5 * dance_similarity) + (0.5 * acoustic_similarity) where categorical values can just be 1 or 0 if they match of now, the numerical values and normalized then we calculate similarity by doing 1-abs(user i - song i) for feature i. Bias is introduced in that we genre the most, and also we don't considered artist. 

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Experiments You Tried

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

I tested basic profiles like High-Energy Pop and Chill-Lofi, which resulted in highly scored songs. This is because the genres/moods matched, and the other values were also very similar. The other thing I tested was high-energy moody using high energy and high valence while being sad/moody. And still it gives me moody songs because the genre/moods match, despite the varying similarity in the valence/energy. For Jazz, but Loud profile with high energy, we got songs that mary both genre and mood, ignoring how dissimilar the energy and tempo are. 

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

The system prioritizes genre and mood. This menans even if there is alternating in the mood/genre over the other numerical features, the system will prioritize genre/mood even if the energy might be giving conflicting information. Another limitation is the small dataset. The dataset is not diverse nor is it extensive, so even if no song is close to user preference, it will still recommend it because we need k songs. We need a diverse dataset that covers all the various possible values of features. 
---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this

I learned about the different ways recommendor systems are developed: collaborative, content-based, or hybrid. I learned how complex these systems can get because of how many different possibilities there are with song features and the values of those features, and the need for a large and diverse dataset. I definitely think recommendation is a very complex task that utilizes many different features and lots of thorough testing and fine-tuning to improve. 

---

## 7. `model_card_template.md`

Combines reflection and model card framing from the Module 3 guidance. :contentReference[oaicite:2]{index=2}  

```markdown
# 🎧 Model Card - Music Recommender Simulation

## 1. Model Name

Give your recommender a name, for example:

> VibeFinder 1.0

---

## 2. Intended Use

- What is this system trying to do
- Who is it for

Example:

> This model suggests 3 to 5 songs from a small catalog based on a user's preferred genre, mood, and energy level. It is for classroom exploration only, not for real users.

---

## 3. How It Works (Short Explanation)

Describe your scoring logic in plain language.

- What features of each song does it consider
- What information about the user does it use
- How does it turn those into a number

Try to avoid code in this section, treat it like an explanation to a non programmer.

---

## 4. Data

Describe your dataset.

- How many songs are in `data/songs.csv`
- Did you add or remove any songs
- What kinds of genres or moods are represented
- Whose taste does this data mostly reflect

---

## 5. Strengths

Where does your recommender work well

You can think about:
- Situations where the top results "felt right"
- Particular user profiles it served well
- Simplicity or transparency benefits

---

## 6. Limitations and Bias

Where does your recommender struggle

Some prompts:
- Does it ignore some genres or moods
- Does it treat all users as if they have the same taste shape
- Is it biased toward high energy or one genre by default
- How could this be unfair if used in a real product

---

## 7. Evaluation

How did you check your system

Examples:
- You tried multiple user profiles and wrote down whether the results matched your expectations
- You compared your simulation to what a real app like Spotify or YouTube tends to recommend
- You wrote tests for your scoring logic

You do not need a numeric metric, but if you used one, explain what it measures.

---

## 8. Future Work

If you had more time, how would you improve this recommender

Examples:

- Add support for multiple users and "group vibe" recommendations
- Balance diversity of songs instead of always picking the closest match
- Use more features, like tempo ranges or lyric themes

---

## 9. Personal Reflection

A few sentences about what you learned:

- What surprised you about how your system behaved
- How did building this change how you think about real music recommenders
- Where do you think human judgment still matters, even if the model seems "smart"

