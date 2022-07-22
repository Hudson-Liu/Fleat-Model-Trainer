# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 20:53:27 2022

@author: Joe Mama
"""

import keras
import cv2
import numpy as np
import pickle
import tensorflow as tf
from keras.utils import to_categorical

# TODO Eventually, this whole program should be converted into a class, and these variables should be set in the driver program (__main__.py) that runs both networkv2 and dataset generator
# we'll also haev a cool __version__.py file
num_images_main = 100
num_images_sub = 1000  # change to none when you have more memory
resolution = [224, 224]


class PreProcessor:
    """An image preprocessor, used to preprocess all the images in a batch"""

    # TODO This should also eventually go in __main__.py
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "

    def __init__(self, num_images, resolution, by_character):
        self.num_images = num_images
        self.resolution = resolution
        self.by_character = by_character

    # The only part that needs to change to convert this from somethign that only takes keras datasets to something that can accept any input dataset is importing image files and converting them into a np array and also grayscaling
    def preprocess_images(self, x, y, rgb):
        # Splice images
        x = x[0:self.num_images]
        y = y[0:self.num_images]

        # Scales images to 64x64 resolution and makes image greyscale
        x_resized = []
        for img in x:
            # if it's enlargening
            if img.shape[0] <= self.resolution[0] and img.shape[1] <= self.resolution[1]:
                interpolation = cv2.INTER_CUBIC
            else:  # if it's downscaling
                interpolation = cv2.INTER_AREA
            resized = cv2.resize(img, dsize=(
                self.resolution[0], self.resolution[1]), interpolation=interpolation)
            if rgb:
                resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            x_resized.append(resized)

        # Normalize
        x_resized = np.array(x_resized)
        x_resized = x_resized.astype("float32") / 255.0

        # Make sure images have shape (224, 224, 1)
        x_resized = np.expand_dims(x_resized, -1)

        # One hot encoding labels
        if self.by_character:
            y_str = [str(i) for i in y] #Character-by-character
            y_num = []
            for ind_x, answer in enumerate(y_str):
                temp = []
                for ind_y, letter in enumerate(answer):
                    one_hot = np.zeros(shape=(len(self.alphabet)))
                    for ind_z, encode in enumerate(self.alphabet):
                        if letter == encode:
                            one_hot[ind_z] = 1
                            break
                    # each individual character's one hot encoding
                    temp.append(one_hot)
                y_num.append(np.array(temp))  # each answer's one hot encoding
            y_num = np.array(y_num)
        else:
            y_num = np.array(to_categorical(y)) #entire label

        dataset = [x_resized, y_num]
        return dataset


main_preproc = PreProcessor(num_images_main, resolution, True)
sub_preproc = PreProcessor(num_images_sub, resolution,  False)
datasets_main = []
datasets_sub = []

# Fashion MNIST
(train_img, train_labels), (test_img,
                            test_labels) = tf.keras.datasets.fashion_mnist.load_data()
imgs = np.concatenate((train_img, test_img), axis=0)
labels = np.concatenate((train_labels, test_labels), axis=0)
datasets_main.append(main_preproc.preprocess_images(imgs, labels, False))
datasets_sub.append(sub_preproc.preprocess_images(imgs, labels, False))

# CIFAR 10
(train_img, train_labels), (test_img,
                            test_labels) = tf.keras.datasets.cifar10.load_data()
imgs = np.concatenate((train_img, test_img), axis=0)
labels = np.concatenate((train_labels, test_labels), axis=0)
datasets_main.append(main_preproc.preprocess_images(imgs, labels, True))
datasets_sub.append(sub_preproc.preprocess_images(imgs, labels, True))

# CIFAR 100
(train_img, train_labels), (test_img,
                            test_labels) = tf.keras.datasets.cifar100.load_data()
imgs = np.concatenate((train_img, test_img), axis=0)
labels = np.concatenate((train_labels, test_labels), axis=0)
datasets_main.append(main_preproc.preprocess_images(imgs, labels, True))
datasets_sub.append(sub_preproc.preprocess_images(imgs, labels, True))

# MNIST
(train_img, train_labels), (test_img, test_labels) = tf.keras.datasets.mnist.load_data()
imgs = np.concatenate((train_img, test_img), axis=0)
labels = np.concatenate((train_labels, test_labels), axis=0)
datasets_main.append(main_preproc.preprocess_images(imgs, labels, False))
datasets_sub.append(sub_preproc.preprocess_images(imgs, labels, False))

# Save all datasets
with open("datasets_main", "wb") as fp:  # Pickling
    pickle.dump(datasets_main, fp)

with open("datasets_sub", "wb") as fp:  # Pickling
    pickle.dump(datasets_sub, fp)
