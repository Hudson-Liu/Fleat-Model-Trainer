# -*- coding: utf-8 -*-
"""
Created Monday 7/20/2022 12:56 PM

Works only on CNN Classification-type problems

@author: joebama
"""
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.optimizers import Adam
import pickle

batch_size = 8
EPOCHS = 10
#INVALID_LOSS = 1000  #loss for an invalid run
#num_images_sub = 1000  # change to none when you have more memory
resolution = [224, 224]

def create_model(output_size):
    model = Sequential()
    model.add(Conv2D(input_shape=(224,224,1),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    #model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=2,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    """
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    """
    model.add(Flatten())
    model.add(Dense(units=100,activation="relu"))
    model.add(Dense(units=100,activation="relu"))
    model.add(Dense(units=output_size, activation="sigmoid"))
    opt = Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model

with open("datasets_sub", "rb") as fp:
    datasets_sub = pickle.load(fp)

for dataset in datasets_sub:
    output_size = dataset[1].shape[1]
    print(output_size)
    model = create_model(output_size)
    hist = model.fit(x=dataset[0], y=dataset[1], batch_size=batch_size, epochs=EPOCHS)
    loss = hist.history
    break

import matplotlib.pyplot as plt
plt.plot(hist.history["acc"])
plt.plot(hist.history['val_acc'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()
