# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 20:53:27 2022

Proof-of-concept PreProcessor
If the network can successfully learn from this small batch of data, even if it's just overfitting, then
it's promising enough to move onto a larger dataset

@author: Joe Mama
"""

import keras
import cv2
import numpy as np

# Eventually, this whole program should be converted into a class, and these variables should be set in the driver program (__main__.py) that runs both networkv2 and dataset generator
# we'll also haev a cool __version__.py file
num_images = 100
train_test_split = 0.8

# PreProcess images (splice dataset, resize, convert to black and white, turn labels into string, and save all using numpy.savez)


class PreProcessor:
    """An image preprocessor, used to preprocess all the images in a batch"""

    def __init__(self, num_images, train_test_split):
        self.num_images = num_images
        self.train_test_split = train_test_split

    # The only part that needs to change to convert this from somethign that only takes keras datasets to something that can accept any input dataset is importing image files and converting them into a np array and also grayscaling
    def preprocess_images(self, x_train, y_train, x_test, y_test, rgb):
        # Splice images
        x_train = x_train[0:int(num_images * train_test_split)]
        x_test = x_test[0:int(num_images - (num_images*train_test_split))]

        # Scales images to 64x64 resolution and makes image greyscale
        train_resized = []
        for img in x_train:
            if img.shape[0] <= 64 and img.shape[1] <= 64:  # if it's enlargening
                interpolation = cv2.INTER_CUBIC
            else:  # if it's downscaling
                interpolation = cv2.INTER_AREA
            resized = cv2.resize(img, dsize=(
                64, 64), interpolation=interpolation)
            if rgb:
                resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            train_resized.append(resized)

        test_resized = []
        for img in x_test:
            if img.shape[0] <= 64 and img.shape[1] <= 64:  # if it's enlargening
                interpolation = cv2.INTER_CUBIC
            else:  # if it's downscaling
                interpolation = cv2.INTER_AREA
            resized = cv2.resize(img, dsize=(
                64, 64), interpolation=interpolation)
            if rgb:
                resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            test_resized.append(resized)

        # Normalize
        train_resized = np.array(train_resized)
        test_resized = np.array(test_resized)
        train_resized = train_resized.astype("float32") / 255.0
        test_resized = test_resized.astype("float32") / 255.0

        # Make sure images have shape (28, 28, 1)
        train_resized = np.expand_dims(train_resized, -1)
        test_resized = np.expand_dims(test_resized, -1)

        # Convert labels into String
        y_train_str = [str(i) for i in y_train]
        y_test_str = [str(i) for i in y_test]

        dataset = [train_resized, np.array(
            y_train_str), test_resized, np.array(y_test_str)]
        return dataset


preprocessor = PreProcessor(num_images, train_test_split)
datasets = []

# Fashion MNIST
(train_img, train_labels), (test_img,
                            test_labels) = keras.datasets.fashion_mnist.load_data()
datasets.append(preprocessor.preprocess_images(
    train_img, train_labels, test_img, test_labels, False))

# CIFAR 10
(train_img, train_labels), (test_img,
                            test_labels) = keras.datasets.cifar10.load_data()
datasets.append(preprocessor.preprocess_images(
    train_img, train_labels, test_img, test_labels, True))

# CIFAR 100
(train_img, train_labels), (test_img,
                            test_labels) = keras.datasets.cifar100.load_data()
datasets.append(preprocessor.preprocess_images(
    train_img, train_labels, test_img, test_labels, True))

# MNIST
(train_img, train_labels), (test_img, test_labels) = keras.datasets.mnist.load_data()
datasets.append(preprocessor.preprocess_images(
    train_img, train_labels, test_img, test_labels, False))
