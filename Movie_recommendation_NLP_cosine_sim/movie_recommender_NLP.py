import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from movie_data_processing import pre_process_movie_data
from sklearn.model_selection import train_test_split
from joblib import dump, load

#def NLP_cosine_similarity():
# get the processed data from process_movie_data.py
processed_movie_data = pre_process_movie_data("movies.csv")
X = processed_movie_data['genres']
y = processed_movie_data['title']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2)

# Instantiating and generating the count matrix
count_vector = CountVectorizer()
count_matrix = count_vector.fit_transform(processed_movie_data['genres'])
  
# list I will use in the function to match the indexes
indices = pd.Series(processed_movie_data.title)
# generating the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)
#print(cosine_sim)

def recommend_movie(title, cosine_sim):
    recommended_movies = []
    # gettin the index of the movie that matches the title
    index = indices[indices == title].index[0]
    #print(index)
    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[index]).sort_values(ascending = False)
    # getting the indexes of the 10 most similar movies
    top_10_movies = list(score_series.iloc[1:11].index)
    
    # populating the list with the titles of the best 10 matching movies
    for i in top_10_movies:
        recommended_movies.append(list(processed_movie_data.title)[i])
        
    return recommended_movies

print(recommend_movie('Avatar (2009)', cosine_sim))


