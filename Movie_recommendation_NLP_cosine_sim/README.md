# About
This code source contains the recommendation system using **Natural language processing & Cosine similarity**
### Code Description
* movie_data_processing.py file contains all the data processing where movie data are cleaned from (removed NaN and no genre given) to any movie.
* movie_recommender_NLP.py file contains the cosine_similarity model after running this file we get related movies based on content recommendation system.
## Query
```javascript
Query = recommend_movie('Avatar (2009)', cosine_sim)
```
## Output

```javascript

['After Earth (2013)', 'Star Trek Into Darkness (2013)', 'Superman Returns (2006)', 'Avatar (2009)', 
'Tron: Legacy (2010)', 'Star Wars: Episode II - Attack of the Clones (2002)', 
'Captain America: The Winter Soldier (2014)', 'John Carter (2012)', 'Avengers, The (2012)', 'Spider-Man 2 (2004)']
```
