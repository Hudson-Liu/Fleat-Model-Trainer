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
import math
import random
import keras_tuner as kt
import config

class DatasetGenerator(tf.keras.utils.Sequence):
    cursor = [0, 0]
    
    def __init__(self, datasets, dataset_identifiers, subnet_data, y, BATCH_SIZE, DATASET_SEGMENTS, NUM_IMAGES):
        """Initializes all the necessary variables from ModelTrainer"""
        self.datasets = datasets
        self.dataset_identifiers = dataset_identifiers
        self.subnet_data = subnet_data
        self.y = y
        self.BATCH_SIZE = BATCH_SIZE
        self.DATASET_SEGMENTS = DATASET_SEGMENTS
        self.NUM_IMAGES = NUM_IMAGES
        
    def __len__(self):
        """Returns the number of batches per sequence"""
        return math.ceil((len(self.dataset_identifiers) * self.DATASET_SEGMENTS) / self.BATCH_SIZE)
    
    def __getitem__(self, idx):
        """Ignores the index value it's handed and instead uses a cursor to keep track of progress"""
        dataset_slices = []
        subnet_instances = [] 
        y_instances = []
        counter = 0
        for index, identifier in enumerate(self.dataset_identifiers[self.cursor[0]:]): #For every dataset unique pair
            for slice_ in range(self.cursor[1], self.DATASET_SEGMENTS): #For every segment within the dataset
                dataset = self.datasets[identifier]
                dataset_slices.append(dataset[slice_ * self.NUM_IMAGES:(slice_ * self.NUM_IMAGES) + self.NUM_IMAGES])
                subnet_instances.append(self.subnet_data[index])
                y_instances.append(self.y[index])
                if counter == self.BATCH_SIZE - 1:
                    self.cursor[0] = index + 1 if slice_ == self.DATASET_SEGMENTS else index #If the next available segment is in the next dataset, save the cursor on the next dataset and not the current
                    self.cursor[1] = (slice_ + 1) % self.DATASET_SEGMENTS #without the plus one, it'd reread the current selection, and the modulo prevents it from going out of bounds
                    return [np.array(dataset_slices), np.array(subnet_instances)], np.array(y_instances)
                counter += 1
                
    def on_epoch_end(self):
        """Resets the cursor at the end of every epoch"""
        self.cursor[0] = 0
        self.cursor[1] = 0
        
class ModelTrainer():
    directory = "/kaggle/input/dataset-but-with-sub-dataset-data/" #for kaggle
    #directory = "" #for local
    NUM_IMAGES = 100  # images per timestep
    TRAIN_TEST_SPLIT = 0.2
    BATCH_SIZE = 2 #Batch size of 1 will not work for whatever reaosn
    NUM_OPTIMIZERS = 8 # Number of unique optimizers being used in "Optimized_Parameters.py"
    DATASET_SEGMENTS = math.floor(config.NUM_IMAGES_MAIN / NUM_IMAGES)
    
    def create_model(self, hp):
        start = time.time()
        print("Started creating model")
        
        #Image embedding
        img_input = keras.Input(shape=(self.NUM_IMAGES, config.RESOLUTION[0], config.RESOLUTION[1], 1))# shape=(timesteps,resolution,resolution,rgb channels)
        x = TimeDistributed(Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu'))(img_input) #Full scale model has 64 filters for both
        x = TimeDistributed(Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu'))(x)
        x = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2))(x)
        
        #Fine tune VGG16
        filters1 = hp.Int('filters1', min_value=8, max_value=128, step=8)
        filters2 = hp.Int('filters2', min_value=8, max_value=128, step=8)
        filters3 = hp.Int('filters3', min_value=8, max_value=128, step=8)
        filters4 = hp.Int('filters4', min_value=8, max_value=128, step=8)
        
        #VGG16
        filters_convs = [(filters1, 2), (filters2, 3), (filters3, 3), (filters4, 3)] #Full scale model, my computer doesn't have enough GPU memory for it
        #filters_convs = [(8, 2), (8, 3), (8, 3), (8, 3)] #Reduced scale model, easier on GPU memory
        for n_filters, n_convs in filters_convs:
            for _ in range(n_convs):
                x = TimeDistributed(Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same', activation='relu'))(x)
            x = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2))(x)
        x = TimeDistributed(Flatten())(x)
        x = TimeDistributed(Dense(units=50), name='Image_Preprocessing')(x)
        
        #Fine tune Embedding
        lstm_units = hp.Int('lstm', min_value=10, max_value=200, step=10)
        image_embed_units = hp.Int('embed', min_value=10, max_value=200, step=10)
        
        #Image Embedding
        x = LSTM(units=lstm_units)(x)
        x = Dropout(.2)(x)
        x = BatchNormalization()(x)
        dataset_embed = Dense(units=image_embed_units, activation='relu', name='Image_Embed')(x)
        dataset_embed = Dropout(.2)(dataset_embed)
        
        #Fine tune model
        model_embed_units = hp.Int('model_embed', min_value=10, max_value=200, step=10)
        
        # Model properties embedding
        properties_input = keras.Input(shape=(self.NUM_OPTIMIZERS + 1))
        properties_embed = Dense(units=model_embed_units, name='Model_Properties_Embed')(properties_input)
        x = Dropout(.2)(properties_embed)

        # Combines both models
        merge = Concatenate(axis=1, name="Total_Embed")([dataset_embed, properties_embed])
        
        #Fune tune learning rate output size
        output_units = hp.Int('output_dense', min_value=10, max_value=200, step=10)
        
        # learning rate output
        x = Dense(units=output_units, activation='relu')(merge)
        x = Dropout(.2)(x)
        output = Dense(units=1, name='Learning_Rate', activation='relu')(x)

        # This model covers all the inputs and outputs
        model = keras.Model([img_input, properties_input], output)
        
        #Compile model
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate), loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError()])
        
        elapsed_time = time.time() - start
        print(f"Finished creating model || Elapsed time: {elapsed_time}s")
        return model

    def import_data(self):
        """Imports all the data necessary for creating x and y"""
        print("Started importing data")
        start = time.time()
        
        with open(f"{self.directory}datasets", "rb") as fp:
            datasets = pickle.load(fp)
            
        with open(f"{self.directory}model_runs_RUN_0", "rb") as fp:
            true_rates = pickle.load(fp)

        with open(f"{self.directory}model_runs_RUN_1", "rb") as fp:
            true_rates.extend(pickle.load(fp))
            
        with open(f"{self.directory}model_runs_RUN_2", "rb") as fp:
            true_rates.extend(pickle.load(fp))
            
        with open(f"{self.directory}model_runs_RUN_3", "rb") as fp:
            true_rates.extend(pickle.load(fp))
        
        elapsed_time = time.time() - start
        print(f"Finished importing data || Elapsed time: {elapsed_time}s")
        
        return datasets, true_rates

    def get_params(self, true_rates):
        """Takes out the parameter column, normalizes those values, and save min & max for use in the fleat library itself"""
        param_num = [i[0] for i in true_rates]
        normalized_param = [(i - min(param_num))/(max(param_num) - min(param_num)) for i in param_num]
        with open("min_max", "wb") as fp:
            pickle.dump([min(param_num), max(param_num)], fp)
        return normalized_param

    def get_optimizers(self, true_rates):
        """Takes the optimizer column out of true_rates and One Hot Encodes the optimizer"""
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
        return encoded

    def get_identifiers(self, true_rates):
        """Takes the dataset identifiers out of true_rates"""
        return [i[2] for i in true_rates]
    
    def get_y(self, true_rates):
        """Takes the y column out of true_rates & multiplies it by 100 so that the max is 100 and the min is 0.01"""
        SCALE_FACTOR = 1000
        y = [i[3] for i in true_rates]
        y = [i * SCALE_FACTOR for i in y]
        return y

    def concat_vals(self, x, y):
        """Concatenate any two lists together"""
        assert len(x) == len(y)
        concat = []
        for index in range(0, len(x)):
            temp = [x[index]]
            for i in y[index]:
                temp.append(i)
            concat.append(temp)
        return np.array(concat)
    
    def train_model(self):
        """Trains and saves model"""
        #Import data
        datasets, true_rates = self.import_data()
        
        #Time preprocessing
        print("Started preprocessing data")
        start = time.time()
        
        #Shuffle dataset
        self.shuffle(true_rates)
        
        #Extract and preprocess all the elements of true_rates
        param_num = self.get_params(true_rates)
        optimizers = self.get_optimizers(true_rates)
        dataset_identifiers = self.get_identifiers(true_rates)
        y = self.get_y(true_rates)
        
        #Merges param_num and optimizers to form the total data on the sub net
        subnet_data = self.concat_vals(param_num, optimizers)
        
        #Creates train test split for all datasetst
        identifiers_train, identifiers_test = self.train_test_split(dataset_identifiers)
        subnet_train, subnet_test = self.train_test_split(subnet_data)
        y_train, y_test = self.train_test_split(y)
        
        #Time preprocessing
        elapsed_time = time.time() - start
        print(f"Finished preprocessing data || Elapsed time: {elapsed_time}s")
        
        #Tensorboard compatibility
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        
        #Create dataset generators
        train_generator = DatasetGenerator(datasets, identifiers_train, subnet_train, y_train, self.BATCH_SIZE, self.DATASET_SEGMENTS, self.NUM_IMAGES)
        test_generator = DatasetGenerator(datasets, identifiers_test, subnet_test, y_test, self.BATCH_SIZE, self.DATASET_SEGMENTS, self.NUM_IMAGES)
        
        #Tune hyperparameters
        tuner = kt.Hyperband(self.create_model,
                     objective='val_mean_absolute_error',
                     max_epochs=20,
                     factor=3,
                     project_name='intro_to_kt')
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        tuner.search(train_generator, 
                     epochs=10, 
                     batch_size=self.BATCH_SIZE, 
                     validation_data=test_generator, 
                     callbacks=[stop_early])
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
        
        #Train final model
        model = tuner.hypermodel.build(best_hps)
        model.fit(train_generator,
                  epochs=10,
                  batch_size=self.BATCH_SIZE,
                  validation_data = test_generator,
                  callbacks=[tensorboard_callback])
        
        model.save('saved_model/Fleat_model')

    def shuffle(self, a):
        """Shuffles a list"""
        random.shuffle(a)

    def train_test_split(self, a):
        """All inputs should not be duplicated using DATASET_SEGMENTS"""
        ratio = round(self.TRAIN_TEST_SPLIT * len(a))
        a_train = a[ratio:]
        a_test = a[:ratio]
        return a_train, a_test

# This way, imports can use the class w/o running the script
if __name__ == "__main__":
    model_trainer = ModelTrainer()
    model_trainer.train_model()
