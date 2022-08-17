# -*- coding: utf-8 -*-
"""
Created on Wed Aug  10 20:14:12 2022

Hopefully, unlike last time, this actually goes well

@author: Joe Mama
"""

import keras
from keras.layers import TimeDistributed, Conv2D, Dense, MaxPooling2D, Flatten, LSTM, Concatenate, Dropout, BatchNormalization
import pickle
import tensorflow as tf
import time
import numpy as np
import datetime
from tqdm import tqdm

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
x = TimeDistributed(Dense(units=50), name='Image_Preprocessing')(x)
x = LSTM(units=50)(x)
x = Dropout(.2)(x)
x = BatchNormalization()(x)
dataset_embed = Dense(units=50, activation='relu', name='Image_Embed')(x)
x = Dropout(.2)(x)

# Model properties embedding
properties_input = keras.Input(shape=(NUM_OPTIMIZERS + 1))
properties_embed = Dense(units=50, name='Model_Properties_Embed')(properties_input)
x = Dropout(.2)(x)

# Combines both models
merge = Concatenate(axis=1, name="Total_Embed")([dataset_embed, properties_embed])

# learning rate output
x = Dense(units=100, activation='relu')(merge)
x = Dropout(.2)(x)
output = Dense(units=1, name='Learning_Rate', activation='relu')(x)

# This model covers all the inputs and outputs
model = keras.Model([img_input, properties_input], output)

# Import and preprocess data
print("Started importing data")
start = time.time()

#directory = "/kaggle/input/dataset-but-with-sub-dataset-data/" #for kaggle
directory = "" #for local
with open(f"{directory}datasets", "rb") as fp:
    datasets = pickle.load(fp)
    
with open(f"{directory}model_runs_RUN_0", "rb") as fp:
    true_rates = pickle.load(fp)

with open(f"{directory}model_runs_RUN_1", "rb") as fp:
    true_rates.extend(pickle.load(fp))
    
with open(f"{directory}model_runs_RUN_2", "rb") as fp:
    true_rates.extend(pickle.load(fp))
    
with open(f"{directory}model_runs_RUN_3", "rb") as fp:
    true_rates.extend(pickle.load(fp))


#Normalize Parameter numbers
param_num = [i[0] for i in true_rates]
normalized_param = [(i - min(param_num))/(max(param_num) - min(param_num)) for i in param_num]
with open("min_max", "wb") as fp:
    pickle.dump([min(param_num), max(param_num)], fp)


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
encoded = np.array(encoded)

#Create y
y = np.array([i[3] for i in true_rates])

#Match up dataset names to datasets
dataset_types = [i[2] for i in true_rates]
#images = np.array([datasets[l] for l in dataset_types])
encoded2 = []
y2 = []
image_chunks = []
for index, dataset_name in enumerate(tqdm(dataset_types)):
    dataset = datasets[dataset_name]
    
    #This code is disgusting but i'm so tired
    image_chunks_list = [dataset[0:100], dataset[100:200]]
    #image_chunks_list = [dataset[0:100]]
    for image_chunk in image_chunks_list:
        image_chunks.append(image_chunk)
        y2.append(y[index])
        encoded2.append(encoded[index])#continues to append the same row
        
#to numpy arrays
encoded2 = np.array(encoded2)
y2 = np.array(y2)
image_chunks = np.array(image_chunks)

#shuffle
assert len(encoded2) == len(y2) == len(image_chunks)
p = np.random.permutation(len(encoded2))
encoded2 = encoded2[p]
y2 = y2[p]
image_chunks = image_chunks[p]

#Create y & multiplies it by 100 so that the max is 100 and the min is 0.01
#This is helpful since the rest of the model is dealing with values ranging between 0 and 1
#NOTE: When doing postprocessing, the value must be divided by 100
#Lowest metric so far, 0.03 accurcy
y2 *= 1000
ratio = round(TRAIN_TEST_SPLIT * len(y2))
y_train = y2[ratio:]
y_val = y2[:ratio]

#Create x
images_train = image_chunks[ratio:]
encoded_train = encoded2[ratio:]
images_val = image_chunks[:ratio]
encoded_val = encoded2[:ratio]
x_train = [images_train, encoded_train]
x_val = [images_val, encoded_val]

#Record time
elapsed = round(time.time() - start, 3)
print(f"Finished importing data || Elapsed Time: {elapsed}s")

#Tensorboard compatibility
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#Generate x dataset
def generate_x():
    

#Just add some more preprocessing and a model.fit and you'll be done
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError()])
model.fit(x=x_train, y=y_train, epochs=50, batch_size=BATCH_SIZE, validation_data = (x_val, y_val), callbacks=[tensorboard_callback])

model.save('saved_model/Fleat_model')
