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
import pickle

# constants/hyperparameters
num_images = 100
batch_size = 69
resolution = [224, 224]
hnet_pred_vars = 9
anet_pred_vars = 25  # the thing on my whiteboard didnt include a stopping node

# CNN-RNN/CNN-LSTM for processing images and corresponding answers
# Image processing
# shape=(timesteps,resolution,resolution,rgb channels)
img_input = keras.Input(
    shape=(num_images, resolution[0], resolution[1], 1), batch_size=batch_size)
x = TimeDistributed(Conv2D(filters=64, kernel_size=(
    3, 3), padding='same', activation='relu'))(img_input)  # used vgg16 as a reference
x = TimeDistributed(Conv2D(filters=64, kernel_size=(
    3, 3), padding='same', activation='relu'))(x)
x = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2))(x)
filters_convs = [(128, 2), (256, 3), (512, 3), (512, 3)]
for n_filters, n_convs in filters_convs:
    for _ in range(n_convs):
        x = TimeDistributed(Conv2D(filters=n_filters, kernel_size=(
            3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2))(x)
x = TimeDistributed(Flatten())(x)
img_embed = TimeDistributed(Dense(units=1000), name='Image_Preprocessing')(x)

# Answer embedding
# Number of image-answer pairs, characters in answer, single character
answer_input = keras.Input(shape=(num_images, None, 1), batch_size=batch_size)
x = TimeDistributed(LSTM(units=1000))(
    answer_input)
answer_embed = TimeDistributed(
    Dense(units=1000), name='Answer_Preprocessing/Embed')(x)

# Combines both models
merge = Concatenate(axis=2)([img_embed, answer_embed])
x = LSTM(units=100)(merge)
dataset_embed = Dense(units=100, activation='relu', name='Dataset_Embed')(x)

# hnet
x = Dense(units=50)(dataset_embed)
hnet_output = Dense(units=hnet_pred_vars, name='Hyperparameters')(x)

# anet
stopping_node = keras.Input(
    shape=(1), batch_size=batch_size, name='Stopping_Node')

# the concatenated inputs of both the combined embeds and the stopping node
anet_input = Concatenate(axis=1)([dataset_embed, stopping_node])
x = Dense(units=50)(anet_input)
x = Dense(units=50)(x)
anet_output = Dense(units=anet_pred_vars, name='Architecture')(x)

# This model covers all the inputs and outputs
model = keras.Model([img_input, answer_input, stopping_node], [
                    hnet_output, anet_output])
plot_model(model, to_file='HYPAT_Model_Structure.png',
           show_shapes=True, show_layer_names=True)

with open("datasets", "rb") as fp:   # Unpickling
    datasets = pickle.load(fp)
