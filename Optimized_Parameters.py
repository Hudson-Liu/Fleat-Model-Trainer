# -*- coding: utf-8 -*-
"""
Created on Wed Aug  10 20:14:12 2022

Hopefully, unlike last time, this actually goes well

Will save an array of arrays of the following format:
[Number of parameters of model, type of optimizer, best learning rate]
each row represents one model

@author: Joe Mama
"""

import tensorflow as tf
import pickle
from keras.layers import Dense, Flatten
import keras
import keras_tuner
import time

#An advantage of creating a dataset like this, is that a relatively small dataset like this turns
#out to become 128 unique elements just because of how permutations work

#Create instances of each of the following models
models = ["resnet50", "vgg16", "inceptionv3", "mobilenetv2"]

#Create an array of possible optimizers
optimizers = ["SGD", "RMSprop", "Adam", "Adadelta", "Adagrad", "Adamax", "Nadam", "Ftrl"]

#Create an array of datasets in "datasets_optimization"
with open("datasets_optimization", "rb") as fp:
    datasets = pickle.load(fp)

#Generate an array of permutations of each of those 3
model_sets = []
for model in models:
    for optimizer in optimizers:
        for dataset in datasets:
            model_sets.append([model, optimizer, dataset])
#SubSets
RUN = 3 #Has to be either 0, 1, 2, or 3; 0 runs for 4 hours, 1 & 2 runs for 10 hours, 3 runs for 4 hours
if RUN == 0: #18
    model_sets = model_sets[:int(len(model_sets)*(4/28))]
elif RUN == 1: #46
    model_sets = model_sets[int(len(model_sets)*(4/28)):int(len(model_sets)*(14/28))]
elif RUN == 2: #45
    model_sets = model_sets[int(len(model_sets)*(14/28)):int(len(model_sets)*(24/28))]
elif RUN == 3: #19
    model_sets = model_sets[int(len(model_sets)*(24/28)):]

#Create a method that takes in a list of model, optimizer, and dataset, and runs keras tuner to find the best learning rate
best_hps = []
def create_models(model_sets):
    counter = 0
    for set in model_sets:
        modelCreator = ModelCreator(set)
        start = time.time()
        tuner = keras_tuner.Hyperband(
            modelCreator.create_single_model,
            objective="val_accuracy",
            overwrite=True, #Every time it's ran the results are resaved
            max_epochs=10,
            factor=3,
            project_name=f'Keras_Tuner_Results_{counter}'
        )
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        tuner.search(set[2][0], set[2][1], validation_split=0.2, callbacks=[stop_early])
        param_number = modelCreator.get_model_params()
        end = time.time() - start
        print(f"TIME ELAPSED: {end}")
        best_hps.append([param_number, set[1], tuner.get_best_hyperparameters(num_trials=1)[0].get("Learning_Rate")])
        with open(f"model_runs_{counter}", "wb") as fp:
            pickle.dump(best_hps, fp)
        counter += 1
        
class ModelCreator:
    def __init__(self, set):
        self.set = set
        
    def create_single_model(self, hp):
        dataset = self.set[2]
        x = dataset[0]
        y = dataset[1]
        
        model_base = self.name_to_model(self.set[0], x)
        x = Flatten()(model_base.outputs[0])
        x = Dense(units=100)(x)
        output = Dense(units=len(y[0]), activation="relu")(x)
        self.model = keras.Model(inputs=model_base.inputs, outputs=output)
        
        #Tune learning rate only
        hp_learning_rate = hp.Float("Learning_Rate", min_value=1e-4, max_value=1e-1, step=5e-3) #change step size to 1e-3 for better results
        
        optimizer = self.name_to_optimizer(self.set[1], hp_learning_rate)
        self.model.compile(optimizer=optimizer, 
                      loss=keras.losses.CategoricalCrossentropy(from_logits=True), 
                      metrics=['accuracy'])
        
        return self.model
    
    def get_model_params(self):
        return self.model.count_params()
    
    def name_to_optimizer(self, optimizer_name, hp_learning_rate):
        match optimizer_name:
            case "SGD":
                return tf.keras.optimizers.SGD(learning_rate=hp_learning_rate)
            case "RMSprop":
                return tf.keras.optimizers.RMSprop(learning_rate=hp_learning_rate)
            case "Adam":
                return tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
            case "Adadelta":
                return tf.keras.optimizers.Adadelta(learning_rate=hp_learning_rate)
            case "Adagrad":
                return tf.keras.optimizers.Adagrad(learning_rate=hp_learning_rate)
            case "Adamax":
                return tf.keras.optimizers.Adamax(learning_rate=hp_learning_rate)
            case "Nadam":
                return tf.keras.optimizers.Nadam(learning_rate=hp_learning_rate)
            case "Ftrl":
                return tf.keras.optimizers.Ftrl(learning_rate=hp_learning_rate)
    
    def name_to_model(self, model_name, x):
        shape = x.shape[1:]
        match model_name:
            case "resnet50":
                return tf.keras.applications.VGG16(include_top=False, weights=None, input_shape=shape)
            case "vgg16":
                return tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=shape)
            case "inceptionv3":
                return tf.keras.applications.InceptionV3(include_top=False, weights=None, input_shape=shape)
            case "mobilenetv2":
                return tf.keras.applications.MobileNetV2(include_top=False, weights=None, input_shape=shape)

create_models(model_sets)
with open("model_runs", "wb") as fp:
    pickle.dump(best_hps, fp)
    
