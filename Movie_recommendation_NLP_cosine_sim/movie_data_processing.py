import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

def pre_process_movie_data(csv_file):
        
    movie_csv = pd.read_csv(csv_file)
    #print(movie_csv.head(20))
    #print(movie_csv.shape)

    # Drop rows with any (no genres listed) cells
    movie_csv = movie_csv[movie_csv.genres != "(no genres listed)"]
    movie_csv.drop(['movieId'], axis=1, inplace=True)

    # Replace "|" with "" in genres column
    genres_list = []
    for row in movie_csv.genres:
        genres_list.append(row.replace("|", " "))
    
    movie_csv.genres = genres_list
    movie_csv.reset_index(drop=True, inplace=True)
    return movie_csv
