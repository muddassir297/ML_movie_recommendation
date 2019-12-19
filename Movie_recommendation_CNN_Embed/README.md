# About
This code source contains the collaborative Recommendation system using **Embedding CNN**.
### Code Description
* ***movies_CNN.py*** file contains all the data processing and all the veriable which are needed to feed the CNN model are created.
* ***movies_embed_Class.py*** file contains Embedding class which is used in train the model.
* ***movies_model_cnn.py*** contains model logic where models are fed with movies data and the model is trained.
* For Collaborative Recommendation system ***movies.csv*** and ***ratings.csv*** files are are taken from data source and movieId, title, genres and userId, movieId coulmns are considered respectively to feed the model

### Inputs for CNN
```javascript
---------------------------
 No of unique users: 20510
 No of unique movies: 25448
 No of unique genre: 1357
 Minimum rating: 0.5
 Maximum rating: 5.0
---------------------------
```
### Model Summery
```javascript
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, 1)            0
__________________________________________________________________________________________________
input_2 (InputLayer)            (None, 1)            0
__________________________________________________________________________________________________
input_3 (InputLayer)            (None, 1)            0
__________________________________________________________________________________________________
embedding_1 (Embedding)         (None, 1, 50)        1025500     input_1[0][0]
__________________________________________________________________________________________________
embedding_2 (Embedding)         (None, 1, 50)        1272400     input_2[0][0]
__________________________________________________________________________________________________
embedding_3 (Embedding)         (None, 1, 50)        67850       input_3[0][0]
__________________________________________________________________________________________________
reshape_1 (Reshape)             (None, 50)           0           embedding_1[0][0]
__________________________________________________________________________________________________
reshape_2 (Reshape)             (None, 50)           0           embedding_2[0][0]
__________________________________________________________________________________________________
reshape_3 (Reshape)             (None, 50)           0           embedding_3[0][0]
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 150)          0           reshape_1[0][0]
                                                                 reshape_2[0][0]
                                                                 reshape_3[0][0]
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 150)          0           concatenate_1[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 64)           9664        dropout_1[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 64)           0           dense_1[0][0]
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 64)           0           activation_1[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 10)           650         dropout_2[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 10)           0           dense_2[0][0]
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, 10)           0           activation_2[0][0]
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 1)            11          dropout_3[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 1)            0           dense_3[0][0]
__________________________________________________________________________________________________
lambda_1 (Lambda)               (None, 1)            0           activation_3[0][0]
==================================================================================================
Total params: 2,376,075
Trainable params: 2,376,075
Non-trainable params: 0
__________________________________________________________________________________________________
```
### Trained Model

```javascript

Train on 1600000 samples, validate on 400000 samples
Epoch 1/5
1600000/1600000 [==============================] - 100s 62us/step - loss: 0.8464 - accuracy: 0.6148 - val_loss: 0.8155 - val_accuracy: 0.6276
Epoch 2/5
1600000/1600000 [==============================] - 97s 60us/step - loss: 0.7097 - accuracy: 0.7273 - val_loss: 0.7025 - val_accuracy: 0.7339
Epoch 3/5
1600000/1600000 [==============================] - 99s 62us/step - loss: 0.6017 - accuracy: 0.7317 - val_loss: 0.6990 - val_accuracy: 0.7396
Epoch 4/5
1600000/1600000 [==============================] - 99s 62us/step - loss: 0.5936 - accuracy: 0.8050 - val_loss: 0.6550 - val_accuracy: 0.7942
Epoch 5/5
1600000/1600000 [==============================] - 100s 63us/step - loss: 0.5555 - accuracy: 0.8382 - val_loss: 0.6014 - val_accuracy: 0.8224
```
### Testing Model
```javascript
Tesing output from test data
[[3.177727 ]
 [3.8467197]
 [1.988964 ]
 ...
 [3.037895 ]
 [3.8513787]
 [2.345988 ]]
```
