# -*- coding: utf-8 -*-
"""
Created Monday 7/6/2022 12:56 PM

Works only on CNN Classification-type problems

joebama

@author: hudso
"""

import keras
from keras.layers import TimeDistributed, Conv2D, Dense, MaxPooling2D, Flatten, LSTM, Concatenate
from tensorflow.keras.utils import plot_model

# constants/hyperparameters
num_images = 100
batch_size = 69
resolution = [64, 64]
embedding_sizes = [50, 50]  # [Dataset embeddings, feature embeddings]
hnet_pred_vars = 9
anet_pred_vars = 25  # the thing on my whiteboard didnt include a stopping node

# CNN-RNN/CNN-LSTM for processing images and corresponding answers
# Image processing
# shape=(timesteps,resolution,resolution,rgb channels)
img_input = keras.Input(
    shape=(num_images, resolution[0], resolution[1], 3), batch_size=batch_size)
x = TimeDistributed(Conv2D(filters=32, kernel_size=(
    3, 3), padding='same', activation='relu'))(img_input)  # used vgg16 as a reference
x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
img_embed = TimeDistributed(Flatten(), name='Image_Preprocessing')(x)
# Answer embedding
# Number of image-answer pairs, characters in answer, single character
answer_input = keras.Input(shape=(num_images, None, 1), batch_size=batch_size)
x = TimeDistributed(LSTM(units=50))(
    answer_input)  # ^^^one character each timestep
answer_embed = TimeDistributed(
    Dense(units=10), name='Answer_Preprocessing/Embed')(x)
# Combines both models
merge = Concatenate(axis=2)([img_embed, answer_embed])
x = LSTM(units=50)(merge)
dataset_embed = Dense(units=50, activation='relu', name='Dataset_Embed')(x)
#dataset_model = keras.Model([img_input, answer_input], dataset_embed)

# Feature NLP
# Number of image-answer pairs, characters in answer, single character
nlp_input = keras.Input(shape=(None, 1), batch_size=batch_size)
x = LSTM(units=50)(nlp_input)  # ^^^one character each timestep
nlp_embed = Dense(units=50, name='Features_Embed')(x)

# Merge all embeddings
merged_embed = Concatenate(axis=1)([dataset_embed, nlp_embed])

# hnet
x = Dense(units=50)(merged_embed)
hnet_output = Dense(units=hnet_pred_vars, name='Hyperparameters')(x)

# anet
stopping_node = keras.Input(
    shape=(1), batch_size=batch_size, name='Stopping_Node')
# the concatenated inputs of both the combined embeds and the stopping node
anet_input = Concatenate(axis=1)([merged_embed, stopping_node])
x = Dense(units=50)(anet_input)
x = Dense(units=50)(x)
anet_output = Dense(units=anet_pred_vars, name='Architecture')(x)

# This model covers all the inputs and outputs
model = keras.Model([img_input, answer_input, nlp_input,
                    stopping_node], [hnet_output, anet_output])
plot_model(model, to_file='HYPAT_Model_Structure.png',
           show_shapes=True, show_layer_names=True)
