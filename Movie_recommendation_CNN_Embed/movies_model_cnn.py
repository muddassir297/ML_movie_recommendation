import os
import numpy as np
from keras.models import Model
from keras.layers import Input, Reshape, Dot, Add, Activation, Lambda, Concatenate, Dense, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.regularizers import l2
from movies_CNN import n_users, n_movies, n_genres, n_factors, X_test_array, y_train, X_train_array, y_test, min_rating, max_rating
from movies_embed_Class import EmbeddingLayer
from keras.models import load_model

def Recommender_model(n_users, n_movies, n_genres, n_factors, min_rating, max_rating):
    user = Input(shape=(1,))
    u1 = EmbeddingLayer(n_users, n_factors)(user)

    movie = Input(shape=(1,))
    m1 = EmbeddingLayer(n_movies, n_factors)(movie)

    genre = Input(shape=(1,))
    g1 = EmbeddingLayer(n_genres, n_factors)(genre)

    x = Concatenate()([u1, m1, g1])
    x = Dropout(0.05)(x)

    x = Dense(64, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dropout(0.05)(x)

    x = Dense(10, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dropout(0.05)(x)

    x = Dense(1, kernel_initializer='he_normal')(x)
    x = Activation('sigmoid')(x)
    x = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(x)

    model = Model(inputs=[user, movie, genre], outputs=x)
    opt = Adam(lr=0.01)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

    return model

def train_model():
    model_embed = Recommender_model(n_users, n_movies, n_genres, n_factors, min_rating, max_rating)
    model_embed.summary()
    # Model training
    history = model_embed.fit(x=X_train_array, y=y_train, batch_size=1000, epochs=5, verbose=1, validation_data=(X_test_array, y_test))

    model_path = './models/'
    #os.mkdir(model_path)
    model_embed.save(model_path + 'embed_movies_model.h5py', overwrite=True)

train_model()

def test_model():
    model_path = './models/'
    model_rec = load_model(model_path + 'embed_movies_model.h5py')
    y_pred = model_rec.predict(X_test_array)
    return y_pred

#print(test_model())
