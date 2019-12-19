import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data_path = 'data/'
movies_filename = 'movies.csv'
ratings_filename = 'ratings.csv'

ratings_csv_data = pd.read_csv(data_path + ratings_filename)

movies_csv_data = pd.read_csv(data_path + movies_filename)
# 
ratings = ratings_csv_data
movies = movies_csv_data

# Merge ratings and movies csv with common movieId
df_movies_data_all = pd.merge(ratings, movies, left_on='movieId', right_on='movieId', how='left').drop(['timestamp'], axis=1)

# Remove NaN and  (no genres listed) from data set
df_movies_data_all = df_movies_data_all[df_movies_data_all.genres != "(no genres listed)"]
df_movies_data_all.dropna(axis=0, inplace=True)

df_movies_data =  df_movies_data_all[:2000000]

# Making non sequential users Id and movies Id in to sequential format to feed into the model with the help of LebelEncoder()
user_enc = LabelEncoder()
df_movies_data['user'] = user_enc.fit_transform(df_movies_data['userId'].values)
n_users = df_movies_data['user'].nunique()

genre_enc = LabelEncoder()
df_movies_data['genre'] = genre_enc.fit_transform(df_movies_data['genres'].values)
n_genres = df_movies_data['genre'].nunique()

item_enc = LabelEncoder()
df_movies_data['movie'] = item_enc.fit_transform(df_movies_data['movieId'].values)
n_movies = df_movies_data['movie'].nunique()

title_enc = LabelEncoder()
df_movies_data['title'] = title_enc.fit_transform(df_movies_data['title'].values)
n_title = df_movies_data['title'].nunique()

# creating min max rating on the basis of unique users and movies
df_movies_data['rating'] = df_movies_data['rating'].values.astype(np.float32)
min_rating = min(df_movies_data['rating'])
max_rating = max(df_movies_data['rating'])

n_users, n_genres, n_movies, min_rating, max_rating

print("---------------------------")
print(" No of unique users: {0}\n No of unique movies: {1}\n No of unique genre: {2}\n Minimum rating: {3}\n Maximum rating: {4}"
    .format(n_users, n_movies, n_genres, min_rating, max_rating))
print("---------------------------")

#Train test data split
X = df_movies_data[['user', 'movie', 'genre']].values
y = df_movies_data['rating'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

#No of factors defind per user/movie, same for both the users and the movies

n_factors = 50

# Turn users and movies in the separate arrays to feed to the keras model as distict inputs
X_train_array = [X_train[:, 0], X_train[:, 1], X_train[:, 2]]
X_test_array = [X_test[:, 0], X_test[:, 1], X_test[:, 2]]
