# -*- coding: utf-8 -*-
"""
Created on Wed Aug  10 20:14:12 2022

Hopefully, unlike last time, this actually goes well

@author: Joe Mama
"""

import keras
from keras.layers import TimeDistributed, Conv2D, Dense, MaxPooling2D, Flatten, LSTM, Concatenate
import pickle
import tensorflow as tf
import time
import numpy as np

# CNN-RNN/CNN-LSTM for processing images and corresponding answers
# Turn this into a global variable thing later
num_images = 100  # images per timestep
resolution = [224, 224]
TRAIN_TEST_SPLIT = 0.2
alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
BATCH_SIZE = 1
NUM_OPTIMIZERS = 8 # Number of unique optimizers being used in "Optimized_Parameters.py"

# Image processing
# shape=(timesteps,resolution,resolution,rgb channels)
img_input = keras.Input(shape=(num_images, resolution[0], resolution[1], 1))
x = TimeDistributed(Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu'))(img_input) #Full scale model has 64 filters for both
x = TimeDistributed(Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu'))(x)
x = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2))(x)
#filters_convs = [(128, 2), (256, 3), (512, 3), (512, 3)] #Full scale model, my computer doesn't have enough GPU memory for it
filters_convs = [(8, 2), (8, 3), (8, 3), (8, 3)] #Reduced scale model, easier on GPU memory
for n_filters, n_convs in filters_convs:
    for _ in range(n_convs):
        x = TimeDistributed(Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2))(x)
x = TimeDistributed(Flatten())(x)
x = TimeDistributed(Dense(units=100), name='Image_Preprocessing')(x)
x = LSTM(units=100)(x)
dataset_embed = Dense(units=100, activation='relu', name='Image_Embed')(x)

# Model properties embedding
properties_input = keras.Input(shape=(NUM_OPTIMIZERS + 1))
properties_embed = Dense(units=100, name='Model_Properties_Embed')(properties_input)

# Combines both models
merge = Concatenate(axis=1, name="Total_Embed")([dataset_embed, properties_embed])

# learning rate output
x = Dense(units=50, activation='relu')(merge)
output = Dense(units=1, name='Learning_Rate', activation='relu')(x)

# This model covers all the inputs and outputs
model = keras.Model([img_input, properties_input], output)

# Import and preprocess data
print("Started importing data")
start = time.time()

directory = "/kaggle/input/dataset-but-with-sub-dataset-data/" #for kaggle
#directory = "" #for local
with open(f"{directory}datasets", "rb") as fp:
    datasets = pickle.load(fp)
    
with open(f"{directory}model_runs_RUN_0", "rb") as fp:
    true_rates = pickle.load(fp)

#Normalize Parameter numbers
param_num = [i[0] for i in true_rates]
normalized_param = [(i - min(param_num))/(max(param_num) - min(param_num)) for i in param_num]

#One Hot Encodes the optimizer
all_optimizers = [i[1] for i in true_rates]
KNOWN_OPTIMIZERS = ["SGD", "RMSprop", "Adam", "Adadelta", "Adagrad", "Adamax", "Nadam", "Ftrl"]
encoded = []
for i in all_optimizers:
    found = -1
    for index, j in enumerate(KNOWN_OPTIMIZERS):
        if i == j:
            found = index
            break
    if found == -1:
        raise ValueError("Invalid optimizer cannot be One-Hot-Encoded")
    else:
        onehot = [0 for _ in KNOWN_OPTIMIZERS]
        onehot[found] = 1
        encoded.append(onehot)

#Concatenates encoded with parameter numbers
assert len(encoded) == len(normalized_param)
for index, element in enumerate(encoded):
    element.append(normalized_param[index])

#Concatenates all the features together to produce "x"
dataset_types = [i[2] for i in true_rates]
images = np.array([datasets[l] for l in dataset_types])
encoded = np.array(encoded)

#Create x and y
x = [images, encoded]
y = np.array([i[3] for i in true_rates])

elapsed = round(time.time() - start, 3)
print(f"Finished importing data || Elapsed Time: {elapsed}s")

#Just add some more preprocessing and a model.fit and you'll be done
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
model.fit(x=x, y=y, epochs=10, batch_size=BATCH_SIZE)
