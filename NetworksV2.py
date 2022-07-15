# -*- coding: utf-8 -*-
"""
Created Monday 7/6/2022 12:56 PM

Works only on CNN Classification-type problems

@author: joebama
"""

import keras
from keras.layers import TimeDistributed, Conv2D, Dense, MaxPooling2D, Flatten, LSTM, Concatenate
from tensorflow.keras.utils import plot_model
import pickle
import tqdm
import tensorflow as tf
import time

# constants/hyperparameters
# TODO: MAKE THEM ALL CAPS SINCE THEYRE CONSTANTS
batch_size = 2
epochs = 10
train_test_split = 0.25


class ArchitectureNet(keras.layers.Layer):  # This will make it possible to repeat this layer over and over
    max_layers = 15

    def __init__(self, anet_pred_vars, **kwargs):
        super().__init__()
        self.anet_pred_vars = anet_pred_vars

        self.concat = Concatenate(axis=1)
        self.dense1 = Dense(units=50, activation='relu')
        self.dense2 = Dense(units=50, activation='relu')
        self.anet_output = Dense(units=self.anet_pred_vars, name='Architecture')
        self.stopping_node = Dense(units=1, activation='sigmoid')

    def call(self, dataset_embed):
        outputs = []
        stopping_node = 0
        num_layers = 0
        input_batch_size = dataset_embed.numpy().shape[0]
        anet_output = tf.zeros([input_batch_size, self.anet_pred_vars], tf.float32)
        while stopping_node < 0.5 and num_layers < self.max_layers:
            x = self.concat([anet_output, dataset_embed])
            x = self.dense1(x)
            x = self.dense2(x)
            anet_output = self.anet_output(x)
            outputs.append(anet_output)

            stop_node_output = self.stopping_node(x)
            stopping_node = stop_node_output.numpy()[0][0]
            num_layers += 1
            # print(stopping_node)
        return outputs

    def get_config(self):
        base_config = super(ArchitectureNet, self).get_config()
        base_config['output_dim'] = self.anet_pred_vars
        return base_config


class StructureModel(keras.Model):
    """My eyes almost hurt as much as my brain"""
    num_images = 100  # images per timestep
    resolution = [224, 224]
    hnet_pred_vars = 9
    anet_pred_vars = 25
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "

    def __init__(self):
        super().__init__()

        # Defining the layers for Model.summary() to work
        self.conv1 = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        self.conv2 = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        self.pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2))

        filters_convs = [(128, 2), (256, 3), (512, 3), (512, 3)]
        self.vgg16layers = []
        for n_filters, n_convs in filters_convs:
            for _ in range(n_convs):
                self.vgg16layers.append(TimeDistributed(Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same', activation='relu')))
            self.vgg16layers.append(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2)))
        self.flatten = TimeDistributed(Flatten())
        self.dense1 = TimeDistributed(Dense(units=1000), name='Image_Preprocessing')

        self.answerLSTM = TimeDistributed(LSTM(units=500))
        self.dense2 = TimeDistributed(Dense(units=1000), name='Answer_Preprocessing/Embed')

        self.concat1 = Concatenate(axis=2)
        self.embedLSTM = LSTM(units=100)
        self.dense3 = Dense(units=100, activation='relu', name='Dataset_Embed')

        self.dense4 = Dense(units=50)
        self.dense5 = Dense(units=self.hnet_pred_vars, name='Hyperparameters')

        self.anet_layer = ArchitectureNet(self.anet_pred_vars)

    def call(self, inputs):
        images = inputs[0]
        answers = inputs[1]

        # CNN-RNN/CNN-LSTM using VGG-16
        x = self.conv1(images)
        x = self.conv2(x)
        x = self.pool1(x)
        for layer in self.vgg16layers:
            x = layer(x)

        x = self.flatten(x)
        img_embed = self.dense1(x)

        # Answer embedding
        x = self.answerLSTM(answers)  # All answers, shape (100, None, 95)
        answer_embed = self.dense2(x)

        # Combines both models
        merge = self.concat1([img_embed, answer_embed])
        x = self.embedLSTM(merge)
        dataset_embed = self.dense3(x)

        # hnet
        x = self.dense4(dataset_embed)
        hnet_output = self.dense5(x)

        # anet
        anet_output = self.anet_layer(dataset_embed)

        return hnet_output, anet_output

    def compile(self):
        super().compile()


# Import and preprocess data
print("Started importing data")
start = time.time()

with open("datasets", "rb") as fp:
    datasets = pickle.load(fp)

elapsed = round(time.time() - start, 3)
print("Finished importing data || Elapsed Time: " + str(elapsed) + "s")

ratio = int(train_test_split * len(datasets))
val = datasets[:ratio]
train = datasets[ratio:]
if len(val) == 0:  # look at me mom i'm a real programmer
    raise IndexError('List \"x_val\" is empty; \"train_test_split\" is set too small')


def generate_tensors(data, img_or_ans):  # 0 for image, 1 for ans
    # technically the images aren't ragged arrays but for simplicity sake we'll keep them alll as ragged tensors
    start = time.time()

    column = [i[img_or_ans] for i in data]
    tensor_data = tf.ragged.constant(column)
    tensor_data = tensor_data.to_tensor()
    tensor_dataset = tf.data.Dataset.from_tensor_slices(tensor_data)

    time_taken = round(time.time() - start, 3)
    return tensor_dataset, time_taken


print("Started creating tensors")
start = time.time()

train_img, time_taken = generate_tensors(train, 0)
print("- Training Image Tensors Created || Elapsed Time: " + str(time_taken) + "s")
train_ans, time_taken = generate_tensors(train, 1)
print("- Training Answer Tensors Created || Elapsed Time: " + str(time_taken) + "s")
val_img, time_taken = generate_tensors(val, 0)
print("- Testing Image Tensors Created || Elapsed Time: " + str(time_taken) + "s")
val_ans, time_taken = generate_tensors(val, 1)
print("- Testing Answer Tensors Created || Elapsed Time: " + str(time_taken) + "s")

elapsed = round(time.time() - start, 3)
print("Finished creating tensors || Total Time: " + str(elapsed) + "s")

# TODO: Test if CIFAR 100 dataset (which has variable length answers) will work
train_img_b = train_img.batch(batch_size)  # b for batched
train_ans_b = train_ans.batch(batch_size)
train_dataset = zip(train_img_b, train_ans_b)

structuremodel = StructureModel()

for (x1, x2) in train_dataset:
    hnet_output, anet_output = structuremodel([x1, x2])
