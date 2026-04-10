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
- The recommender is designed to help recommened songs to a user given the users preferences such as genre, mood, energy, valence, acousticness, tempo, dancebility. It will use a points based system where a genre match= 2 points, mood match=1.5 point, energy=1 point, tempo=1 point, valence=1 point, dance=0.5, acoustic=0.5. This is designed for classroom exploration as it is a basic prototype, but cannot be generalized as it's a small dataset. 
---

## 3. How the Model Works  

Explain your scoring approach in simple language.  

Prompts:  

- What features of each song are used (genre, energy, mood, etc.)  
- What user preferences are considered  
- How does the model turn those into a score  
- What changes did you make from the starter logic  
- Avoid code here. Pretend you are explaining the idea to a friend who does not program.
- We calculate similarity between each song and user preference. To do so, for each song, we find the each feature similarity as the 1-abs(user_pref_feature_i - song_feature_i) for numerical and using 1 OR 0 for genre/mood match. Then each similarity is multiplied by a points based weight system where genre=2 points, mood=1.5 points, energy=1 points, tempo=1 point, valence=1 point, dance=0.5, acoustic=0.5. We sum over the the products and return the score. 
---

## 4. Data  

Describe the dataset the model uses.  

Prompts:  

- How many songs are in the catalog  
- What genres or moods are represented  
- Did you add or remove data  
- Are there parts of musical taste missing in the dataset  
- There are about 20 songs in the dataset. It represents pop, lofi, ambient, jazz, and some other singular songs in other genres. We have moody, happy, chill, relaxed, and more moods as well. I think the songs represented are diverse in moods and genres, but the frequency of them could be increased. Moreover, we can also increase the frequency of the different possible values of the numerical features like energy to get more diverse songs per mood and genre to better generalize for specific user preferences. 
---

## 5. Strengths  

Where does your system seem to work well  

Prompts:  

- User types for which it gives reasonable results  
- Any patterns you think your scoring captures correctly  
- Cases where the recommendations matched your intuition  
- The model does pretty well at finding songs where the mood and genres match. This makes sense as those are given most weight and would increase the score of the song. This also means that if the user prefers songs of a specific mood or genre, it will likely give those songs, even if the other features like energy or valence don't match. And in one way, this is good because songs that match our mood and genre preference should be the ones that get more priority in being recommened. 
---

## 6. Limitations and Bias 

Where the system struggles or behaves unfairly. 

Prompts:  

- Features it does not consider  
- Genres or moods that are underrepresented  
- Cases where the system overfits to one preference  
- Ways the scoring might unintentionally favor some users  
- The system prioritizes genre and mood. This menans even if there is alternating in the mood/genre over the other numerical features, the system will prioritize genre/mood even if the energy might be giving conflicting information. Another limitation is the small dataset. The dataset is not diverse nor is it extensive, so even if no song is close to user preference, it will still recommend it because we need k songs. We need a diverse dataset that covers all the various possible values of features. 
---

## 7. Evaluation  

How you checked whether the recommender behaved as expected. 

Prompts:  

- Which user profiles you tested  
- What you looked for in the recommendations  
- What surprised you  
- Any simple tests or comparisons you ran  
- No need for numeric metrics unless you created some.
- I tested basic profiles like High-Energy Pop and Chill-Lofi, which resulted in highly scored songs. This is because the genres/moods matched, and the other values were also very similar. The other thing I tested was high-energy moody using high energy and high valence while being sad/moody. And still it gives me moody songs because the genre/moods match, despite the varying similarity in the valence/energy. For Jazz, but Loud profile with high energy, we got songs that mary both genre and mood, ignoring how dissimilar the energy and tempo are. 
---

## 8. Future Work  

Ideas for how you would improve the model next.  

Prompts:  

- Additional features or preferences  
- Better ways to explain recommendations  
- Improving diversity among the top results  
- Handling more complex user tastes  
- One limitation is the small dataset. The dataset is not diverse nor is it extensive, so even if no song is close to user preference, it will still recommend it because we need k songs. We need a diverse dataset that covers all the various possible values of features. We could also include the collaborative user interaction data (like likes, comments, shares) to find others who played songs with similar taste. This could better improve our song recommendations. 

---

## 9. Personal Reflection  

A few sentences about your experience.  

Prompts:  

- What you learned about recommender systems  
- Something unexpected or interesting you discovered  
- How this changed the way you think about music recommendation apps  
- I learned about the different ways recommendor systems are developed: collaborative, content-based, or hybrid. I learned how complex these systems can get because of how many different possibilities there are with song features and the values of those features, and the need for a large and diverse dataset. I definitely think recommendation is a very complex task that utilizes many different features and lots of thorough testing and fine-tuning to improve. 