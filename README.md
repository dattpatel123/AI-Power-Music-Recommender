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
# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

Give your model a short, descriptive name.  
Example: **VibeFinder 1.0**  

**Music-Matcher**

---

## 2. Intended Use  

Describe what your recommender is designed to do and who it is for. 

Prompts:  

- What kind of recommendations does it generate  
- What assumptions does it make about the user  
- Is this for real users or classroom exploration  
The recommender is designed to help recommened songs to a user given the users preferences such as genre, mood, energy, valence, acousticness, tempo, dancebility. It will use a points based system where a genre match= 2 points, mood match=1.5 point, energy=1 point, tempo=1 point, valence=1 point, dance=0.5, acoustic=0.5. This is designed for classroom exploration as it is a basic prototype, but cannot be generalized as it's a small dataset. 
---

## 3. How the Model Works  

Explain your scoring approach in simple language.  

Prompts:  

- What features of each song are used (genre, energy, mood, etc.)  
- What user preferences are considered  
- How does the model turn those into a score  
- What changes did you make from the starter logic  

Avoid code here. 

Pretend you are explaining the idea to a friend who does not program.
We calculate similarity between each song and user preference. To do so, for each song, we find the each feature similarity as the 1-abs(user_pref_feature_i - song_feature_i) for numerical and using 1 OR 0 for genre/mood match. Then each similarity is multiplied by a points based weight system where genre=2 points, mood=1.5 points, energy=1 points, tempo=1 point, valence=1 point, dance=0.5, acoustic=0.5. We sum over the the products and return the score. 
---

## 4. Data  

Describe the dataset the model uses.  

Prompts:  

- How many songs are in the catalog  
- What genres or moods are represented  
- Did you add or remove data  
- Are there parts of musical taste missing in the dataset  
There are about 20 songs in the dataset. It represents pop, lofi, ambient, jazz, and some other singular songs in other genres. We have moody, happy, chill, relaxed, and more moods as well. I think the songs represented are diverse in moods and genres, but the frequency of them could be increased. Moreover, we can also increase the frequency of the different possible values of the numerical features like energy to get more diverse songs per mood and genre to better generalize for specific user preferences. 
---

## 5. Strengths  

Where does your system seem to work well  

Prompts:  

- User types for which it gives reasonable results  
- Any patterns you think your scoring captures correctly  
- Cases where the recommendations matched your intuition  

---
The model does pretty well at finding songs where the mood and genres match. This makes sense as those are given most weight and would increase the score of the song. This also means that if the user prefers songs of a specific mood or genre, it will likely give those songs, even if the other features like energy or valence don't match. And in one way, this is good because songs that match our mood and genre preference should be the ones that get more priority in being recommened. 

## 6. Limitations and Bias 

Where the system struggles or behaves unfairly. 

Prompts:  

- Features it does not consider  
- Genres or moods that are underrepresented  
- Cases where the system overfits to one preference  
- Ways the scoring might unintentionally favor some users  

---

The system prioritizes genre and mood. This menans even if there is alternating in the mood/genre over the other numerical features, the system will prioritize genre/mood even if the energy might be giving conflicting information. Another limitation is the small dataset. The dataset is not diverse nor is it extensive, so even if no song is close to user preference, it will still recommend it because we need k songs. We need a diverse dataset that covers all the various possible values of features. 

## 7. Evaluation  

How you checked whether the recommender behaved as expected. 

Prompts:  

- Which user profiles you tested  
- What you looked for in the recommendations  
- What surprised you  
- Any simple tests or comparisons you ran  

No need for numeric metrics unless you created some.

I tested basic profiles like High-Energy Pop and Chill-Lofi, which resulted in highly scored songs. This is because the genres/moods matched, and the other values were also very similar. The other thing I tested was high-energy moody using high energy and high valence while being sad/moody. And still it gives me moody songs because the genre/moods match, despite the varying similarity in the valence/energy. For Jazz, but Loud profile with high energy, we got songs that mary both genre and mood, ignoring how dissimilar the energy and tempo are. 
---

## 8. Future Work  

Ideas for how you would improve the model next.  

Prompts:  

- Additional features or preferences  
- Better ways to explain recommendations  
- Improving diversity among the top results  
- Handling more complex user tastes  


One limitation is the small dataset. The dataset is not diverse nor is it extensive, so even if no song is close to user preference, it will still recommend it because we need k songs. We need a diverse dataset that covers all the various possible values of features. We could also include the collaborative user interaction data (like likes, comments, shares) to find others who played songs with similar taste. This could better improve our song recommendations. 
---
## 9. Personal Reflection  

A few sentences about your experience.  

Prompts:  

- What you learned about recommender systems  
- Something unexpected or interesting you discovered  
- How this changed the way you think about music recommendation apps  

I learned about the different ways recommendor systems are developed: collaborative, content-based, or hybrid. I learned how complex these systems can get because of how many different possibilities there are with song features and the values of those features, and the need for a large and diverse dataset. I definitely think recommendation is a very complex task that utilizes many different features and lots of thorough testing and fine-tuning to improve. 