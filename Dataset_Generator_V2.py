# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 20:53:27 2022

@author: Joe Mama
"""

import cv2
import numpy as np
import pickle
import tensorflow as tf
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import config

class PreProcessor:
    """An image preprocessor, used to preprocess all the images in a batch"""
    
    def __init__(self, num_images, resolution, use_labels):
        self.num_images = num_images
        self.resolution = resolution
        self.use_labels = use_labels

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
        if self.use_labels:
            y_num = np.array(to_categorical(y)) #entire label
            dataset = [x_resized, y_num]
            return dataset
        else:
            return x_resized

if __name__ == "__main__":
    main_preproc = PreProcessor(config.NUM_IMAGES_MAIN, config.RESOLUTION, False)
    sub_preproc = PreProcessor(config.NUM_IMAGES_SUB, config.RESOLUTION,  True)
    datasets_main = {}
    datasets_sub = {}

    # Fashion MNIST
    (train_img, train_labels), (test_img,
                                test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    imgs = np.concatenate((train_img, test_img), axis=0)
    labels = np.concatenate((train_labels, test_labels), axis=0)
    datasets_main.update({"fashion-mnist" : main_preproc.preprocess_images(imgs, labels, False)})
    datasets_sub.update({"fashion-mnist" : sub_preproc.preprocess_images(imgs, labels, False)})

    # CIFAR 10
    (train_img, train_labels), (test_img,
                                test_labels) = tf.keras.datasets.cifar10.load_data()
    imgs = np.concatenate((train_img, test_img), axis=0)
    labels = np.concatenate((train_labels, test_labels), axis=0)
    datasets_main.update({"cifar10" : main_preproc.preprocess_images(imgs, labels, True)})
    datasets_sub.update({"cifar10" : sub_preproc.preprocess_images(imgs, labels, True)})

    # CIFAR 100
    (train_img, train_labels), (test_img,
                                test_labels) = tf.keras.datasets.cifar100.load_data()
    imgs = np.concatenate((train_img, test_img), axis=0)
    labels = np.concatenate((train_labels, test_labels), axis=0)
    datasets_main.update({"cifar100" : main_preproc.preprocess_images(imgs, labels, True)})
    datasets_sub.update({"cifar100" : sub_preproc.preprocess_images(imgs, labels, True)})

    # MNIST
    (train_img, train_labels), (test_img, test_labels) = tf.keras.datasets.mnist.load_data()
    imgs = np.concatenate((train_img, test_img), axis=0)
    labels = np.concatenate((train_labels, test_labels), axis=0)
    datasets_main.update({"mnist" : main_preproc.preprocess_images(imgs, labels, False)})
    datasets_sub.update({"mnist" : sub_preproc.preprocess_images(imgs, labels, False)})

    # Save all datasets
    with open("datasets", "wb") as fp:  # Pickling
        pickle.dump(datasets_main, fp)

    with open("datasets_optimization", "wb") as fp:  # Pickling
        pickle.dump(datasets_sub, fp)
