from movies_KNN import movie_to_idx, movie_user_mat_sparse
from fuzzywuzzy import fuzz
import pickle

def match_movie_title(mapper, fav_movie, verbose=True):

    match_tuple = []
    # get match
    for title, idx in mapper.items():
        ratio = fuzz.ratio(title.lower(), fav_movie.lower())
        if ratio >= 60:
            match_tuple.append((title, idx, ratio))
    # sort
    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    if not match_tuple:
        print('Match not found')
        return
    if verbose:
        print('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))
    return match_tuple[0][1]

def recommend_movie(model_knn, data, mapper, fav_movie, n_recommendations):
        
    # fit
    model_knn.fit(data)
    # get input movie index
    print('You have input movie:', fav_movie)
    idx = match_movie_title(mapper, fav_movie, verbose=True)
    
    print('Recommendation system start to make inference')
    distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations+1)
    
    raw_recommends = \
        sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    # get reverse mapper
    reverse_mapper = {v: k for k, v in mapper.items()}
    # print recommendations
    print('Recommendations for {}:'.format(fav_movie))
    for i, (idx, dist) in enumerate(raw_recommends):
        print('{0}: {1}, with distance of {2}'.format(i+1, reverse_mapper[idx], dist))



# load the model from disk
dir_path = './models/'
loaded_model = pickle.load(open(dir_path+'knn_model_file', 'rb'))
model_knn = loaded_model
my_favorite = 'Iron Man'

# Call function for recommendation
recommend_movie(
    model_knn=model_knn,
    data=movie_user_mat_sparse,
    fav_movie=my_favorite,
    mapper=movie_to_idx,
    n_recommendations=10)