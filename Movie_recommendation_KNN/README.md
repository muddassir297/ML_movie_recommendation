# About
This code source contains the recommendation system using **K Nearest Neighbor**.
### Code Description
* movies_KNN.py file contains all the data processing and training the model upon running the file, model is saved in "models" folder which will be used for testing purposes.
* recommend_movies.py file contains the testing method as well as getting matching movies and recommended movies.
* For this Recommendation system movies.csv and ratings.csv files are are taken from data source and movieId, title, genres and userId, movieId coulmns are considered respectively to feed the model
## Query
```javascript
my_favorite = 'Iron Man'

recommend_movie(
    model_knn=model_knn,
    data=movie_user_mat_sparse,
    fav_movie=my_favorite,
    mapper=movie_to_idx,
    n_recommendations=10)
```
## Output

```javascript

Recommendations for Iron Man:
1: Bourne Ultimatum, The (2007), with distance of 0.4217848777770996
2: Sherlock Holmes (2009), with distance of 0.4190899133682251
3: Inception (2010), with distance of 0.39293038845062256
4: Avatar (2009), with distance of 0.38322633504867554
5: WALL·E (2008), with distance of 0.38314002752304077
6: Star Trek (2009), with distance of 0.37503182888031006
7: Batman Begins (2005), with distance of 0.3701704144477844
8: Iron Man 2 (2010), with distance of 0.37011009454727173
9: Avengers, The (2012), with distance of 0.3579972982406616
10: Dark Knight, The (2008), with distance of 0.3010351061820984
```
