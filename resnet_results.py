# -*- coding: utf-8 -*-
"""
Created Monday 7/20/2022 12:56 PM

Works only on CNN Classification-type problems

@author: joebama
"""
import keras
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
import pickle as pickle
import tensorflow as tf
import matplotlib.pyplot as plt

BATCH_SIZE = 8
EPOCHS = 10
resolution = [224, 224]

def create_model(output_size):
    #Has input of shape (224, 224, 1)
    model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=None, input_shape=(resolution[0], resolution[1], 1))
    x = Flatten()(model.outputs[0])
    x = Dense(units=100)(x)
    output = Dense(units=output_size)(x)
    model = keras.Model(inputs=model.inputs, outputs=output)
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss=keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

with open("datasets_sub", "rb") as fp:
    datasets_sub = pickle.load(fp)

loss_values = []
for dataset in datasets_sub:
    output_size = dataset[1].shape[1]
    model = create_model(output_size)
    hist = model.fit(x=dataset[0], y=dataset[1], batch_size=BATCH_SIZE, epochs=EPOCHS)
    loss_values.append(hist.history['loss'][EPOCHS - 1])

    plt.plot(hist.history['loss'])
    plt.title("model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("MONKE")
    plt.show()

with open("loss", "wb") as fp:
    pickle.dump(loss_values, fp)
