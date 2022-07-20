# -*- coding: utf-8 -*-
"""
Created Monday 7/6/2022 12:56 PM

Works only on CNN Classification-type problems

@author: joebama
"""

import keras
from keras.layers import TimeDistributed, Conv2D, Dense, MaxPooling2D, Flatten, LSTM, Concatenate, Dropout, BatchNormalization, AveragePooling2D
from keras.utils import plot_model
import pickle
import tqdm
import tensorflow as tf
import time
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading


class ArchitectureNet(keras.layers.Layer):  # This will make it possible to repeat this layer over and over
    max_layers = 15

    def __init__(self, anet_pred_vars, **kwargs):
        super().__init__()
        self.anet_pred_vars = anet_pred_vars

        self.concat = Concatenate(axis=1)
        self.dense1 = Dense(units=50, activation='relu')
        self.dense2 = Dense(units=50, activation='relu')
        self.anet_output = Dense(units=self.anet_pred_vars, activation='relu', name='Architecture')
        self.stopping_node = Dense(units=1, activation='sigmoid')

    def call(self, dataset_embed_batch):
        #Loop through all batches
        dataset_embed_batch = dataset_embed_batch.numpy()
        all_outputs = []
        for embed in dataset_embed_batch:
            # Reformat embeds
            embed = np.expand_dims(embed, axis=0)
            tensor_embed = tf.convert_to_tensor(embed)

            # Create layers
            outputs = []
            stopping_node = 0
            num_layers = 0
            anet_output = tf.zeros([1, self.anet_pred_vars], tf.float32)
            while stopping_node < 0.5 and num_layers < self.max_layers:
                x = self.concat([anet_output, tensor_embed])
                x = self.dense1(x)
                x = self.dense2(x)
                anet_output = self.anet_output(x)
                outputs.append(anet_output)

                stop_node_output = self.stopping_node(x)
                stopping_node = stop_node_output.numpy()[0][0]
                num_layers += 1
                # print(stopping_node)
            all_outputs.append(outputs)
        return all_outputs

    def get_config(self):
        base_config = super(ArchitectureNet, self).get_config()
        base_config['output_dim'] = self.anet_pred_vars
        return base_config


class StructureModel(keras.Model):
    """My eyes almost hurt as much as my brain"""
    num_images = 100  # images per timestep
    num_sub_images = 1000
    resolution = [224, 224]
    hnet_pred_vars = 9
    anet_pred_vars = 27
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
        self.dense1 = TimeDistributed(Dense(units=1000, activation='relu'), name='Image_Preprocessing')

        self.answerLSTM = TimeDistributed(LSTM(units=500))
        self.dense2 = TimeDistributed(Dense(units=1000, activation='relu'), name='Answer_Preprocessing/Embed')

        self.concat1 = Concatenate(axis=2)
        self.embedLSTM = LSTM(units=100)
        self.dense3 = Dense(units=100, activation='relu', name='Dataset_Embed')

        self.dense4 = Dense(units=50, activation='relu')
        self.dense5 = Dense(units=self.hnet_pred_vars, activation='relu', name='Hyperparameters')

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

    def custom_loss(self, current_datasets, hnet_output_total, anet_output_total):
        #hyperparameters
        batch_size = 32
        EPOCHS = 10
        INVALID_LOSS = 1000  #loss for an invalid run

        #Anet output is a list of batches of other outputs so no conversion for it
        hnet_output_total = hnet_output_total.numpy()

        for dataset_number, individual_dataset in enumerate(current_datasets): # Separates the batch into individual dataset
            #anet_output_total = 

            # current_dataset = datasets_sub[step] #Selects the current dataset
            input_layer = keras.Input(shape=(self.resolution[0], self.resolution[1], 1))
            x = input_layer #just to get the loop started

            hnet_output = hnet_output_total[dataset_number]
            anet_output = anet_output_total[dataset_number]
            num_optimizer = np.argmax(hnet_output[:8])

            learning_rate = 0.05 * hnet_output[8] #Minimum learning rate is unbounded, max is 0.05
            optimizer = self.convert_optimizer(num_optimizer, learning_rate)

            #Create hidden layers
            for layer in anet_output:
                layer = layer.numpy()
                
                #Categorical
                layer_num = layer[0][:7]
                layer_num = np.argmax(layer_num)
                activation_num = layer[0][7:16]
                activation_num = np.argmax(activation_num)
                padding_num = layer[0][16:18]
                padding_num = np.argmax(padding_num)

                #Numerical
                strides = layer[0][18:20]
                num_nodes = layer[0][20]
                num_filters = layer[0][21]
                kernel_size = layer[0][22:24]
                pool_size = layer[0][24:26]
                dropout_rate = layer[0][26]

                #Create the layer
                try:
                    match layer_num:
                        case 0:
                            activation = self.convert_activation(activation_num)
                            padding_type= self.convert_padding(padding_num)
                            num_filters = int(abs(num_filters))
                            x = Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding=padding_type, activation=activation)(x)
                        case 1:
                            padding_type = self.convert_padding(padding_num)
                            x = MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding_type)(x)
                        case 2:
                            activation = self.convert_activation(activation_num)
                            if num_nodes > 1000:
                                num_nodes = 1000
                            x = Dense(units=num_nodes, activation=activation)(x)
                        case 3:
                            x = Flatten()(x)
                        case 4:
                            dropout_rate = self.convert_dropout(dropout_rate)
                            x = Dropout(rate=dropout_rate)(x)
                        case 5:
                            x = BatchNormalization()(x)
                        case 6:
                            padding_type= self.convert_padding(padding_num)
                            x = AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding_type)(x)
                except ValueError as error:
                    print(error)
                    return INVALID_LOSS #really high loss when it's invalid
            
            # Output layer
            #FOR ALL INSTANCES OF CURRENT_DATASET MAKE ALL THE INDEXES WORK WITH BATCHES (select)
            output_size = individual_dataset[1].shape[1] # Gets encoding size of categorical data
            outputs = Dense(units=output_size)(x)
            model = keras.Model(inputs=input_layer, outputs=outputs, name="subnet")
            tf.keras.utils.plot_model(model, to_file="subnet.png", show_shapes=True)

            #Compiles and trains
            model.compile(
                loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                optimizer=optimizer,
                metrics=["accuracy"],
            )

            #Verifies if model has the right output dimensions and trains
            last_layer_size = model.layers[-1].output_shape
            if last_layer_size != individual_dataset[1].shape:
                return INVALID_LOSS
            try:
                history = model.fit(x = individual_dataset[0], y = individual_dataset[1], batch_size=batch_size, epochs=EPOCHS, validation_split=0.2)
            except:
                print("Something went wrong")
                return INVALID_LOSS

        return history.history['val_loss']
    
    def convert_optimizer(self, optimizer_num, learning_rate):
        match optimizer_num:
            case 0:
                return keras.optimizers.SGD(learning_rate=learning_rate)
            case 1:
                return keras.optimizers.RMSprop(learning_rate=learning_rate)
            case 2:
                return keras.optimizers.Adam(learning_rate=learning_rate)
            case 3:
                return keras.optimizers.Adadelta(learning_rate=learning_rate)
            case 4:
                return keras.optimizers.Adagrad(learning_rate=learning_rate)
            case 5:
                return keras.optimizers.Adamax(learning_rate=learning_rate)
            case 6:
                return keras.optimizers.Nadam(learning_rate=learning_rate)
            case 7:
                return keras.optimizers.Ftrl(learning_rate=learning_rate)
    
    def convert_activation(self, activation_num):
        match activation_num:
            case 0:
                return "relu"
            case 1:
                return "sigmoid"
            case 2:
                return "softmax"
            case 3:
                return "softplus"
            case 4:
                return "softsign"
            case 5:
                return "tanh"
            case 6:
                return "selu"
            case 7:
                return "elu"
            case 8:
                return "exponential"
    
    def convert_padding(self, padding_num):
        #Yes, an if else statement could do this in less lines, but c o n s i s t e n c y
        match padding_num:
            case 0:
                return "valid"
            case 1:
                return "same"
    
    def convert_dropout(self, dropout_num):
        """Manually maps droput to between 0 and 1 using Sigmoid function"""
        a = np.exp(-1 * dropout_num) # z=e^-x
        dropout_rate = 1 / (1 + a)
        return dropout_rate

class ModelTrainer():
    # constants/hyperparameters
    # TODO: MAKE THEM ALL CAPS SINCE THEYRE CONSTANTS
    batch_size = 2
    epochs = 10
    train_test_split = 0.25

    def __init__(self, plot1, canvas, pb, value_label, pb2, value_label2):
        self.plot1 = plot1
        self.canvas = canvas
        self.pb = pb
        self.value_label = value_label
        self.pb2 = pb2
        self.value_label2 = value_label2

    def import_data(self):
        # Import and preprocess data
        print("Started importing data")
        start = time.time()

        with open("datasets_main", "rb") as fp:
            datasets = pickle.load(fp)

        with open("datasets_sub", "rb") as fp:
            self.datasets_sub = pickle.load(fp)

        elapsed = round(time.time() - start, 3)
        print("Finished importing data || Elapsed Time: " + str(elapsed) + "s")

        ratio = int(self.train_test_split * len(datasets))
        val = datasets[:ratio]
        train = datasets[ratio:]
        if len(val) == 0:  # look at me mom i'm a real programmer
            raise IndexError('List \"x_val\" is empty; \"train_test_split\" is set too small')
        
        print("Started creating tensors")
        start = time.time()

        train_img, time_taken = self.generate_tensors(train, 0)
        print("- Training Image Tensors Created || Elapsed Time: " + str(time_taken) + "s")
        train_ans, time_taken = self.generate_tensors(train, 1)
        print("- Training Answer Tensors Created || Elapsed Time: " + str(time_taken) + "s")
        val_img, time_taken = self.generate_tensors(val, 0)
        print("- Testing Image Tensors Created || Elapsed Time: " + str(time_taken) + "s")
        val_ans, time_taken = self.generate_tensors(val, 1)
        print("- Testing Answer Tensors Created || Elapsed Time: " + str(time_taken) + "s")

        elapsed = round(time.time() - start, 3)
        print("Finished creating tensors || Total Time: " + str(elapsed) + "s")

        # b for batched
        self.train_img_b = train_img.batch(self.batch_size)
        self.train_ans_b = train_ans.batch(self.batch_size)
        self.num_batches_per_epoch = int(len(train) / self.batch_size) + 1 if len(train) > 0 else int(len(train) / self.batch_size) 

    def generate_tensors(self, data, img_or_ans):
        """Returns a 0 for image, 1 for answer"""
        start = time.time()

        column = [i[img_or_ans] for i in data]
        tensor_data = tf.ragged.constant(column)
        tensor_data = tensor_data.to_tensor()
        tensor_dataset = tf.data.Dataset.from_tensor_slices(tensor_data)

        time_taken = round(time.time() - start, 3)
        return tensor_dataset, time_taken

    def main_training_loop(self):
        """Main Training Loop: Uses Epochs, train_img_b, train_ans_b, batch_size, and datasets_sub"""
        structuremodel = StructureModel()
        optimizer = keras.optimizers.SGD(learning_rate=0.001)
        epoch_list = []
        loss_array = []
        EPOCH_PERCENT = 100.0 / float(self.epochs)
        STEP_PERCENT = 100.0 / self.num_batches_per_epoch
        print(self.num_batches_per_epoch)
        for epoch in tqdm.tqdm(range(0, self.epochs), desc="Epochs"):
            epoch_list.append(epoch)
            # Iterate over the batches of the dataset.
            train_dataset = zip(self.train_img_b, self.train_ans_b) #it's a generator so it has to be redefined every time
            loss_sum = 0
            self.pb2['value'] = 0
            self.value_label2['text'] = "Current Progress: 0%"
            for step, (train_imgs, train_ans) in enumerate(train_dataset):
                #with tf.GradientTape() as tape:
                    #hnet_output, anet_output = structuremodel([train_imgs, train_ans], training=True)

                    #current_dataset = datasets_sub[step]  # Selects the current dataset
                    #loss_value = structuremodel.custom_loss(current_dataset, hnet_output, anet_output)

                hnet_output, anet_output = structuremodel([train_imgs, train_ans])

                # Compute the loss value for this minibatch.
                a = step * self.batch_size
                b = (step * self.batch_size) + self.batch_size
                if b > len(self.datasets_sub): #catches out of bounds for last batch
                    b = len(self.datasets_sub)
                current_dataset = self.datasets_sub[a:b]  # Selects the current dataset
                loss_value = structuremodel.custom_loss(current_dataset, hnet_output, anet_output)
                loss_sum += loss_value
                
                self.pb2['value'] += STEP_PERCENT
                self.value_label2['text'] = "Current Progress: " + str(self.pb2['value']) + "%"
                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                #grads = tape.gradient(loss_value, model.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                #optimizer.apply_gradients(zip(grads, model.trainable_weights))
            average_loss = float(loss_sum) / self.num_batches_per_epoch
            loss_array.append(average_loss)

            self.plot1.scatter(epoch_list, loss_array)
            self.plot1.plot(epoch_list, loss_array)
            self.canvas.draw()

            self.pb['value'] += EPOCH_PERCENT
            self.value_label['text'] = "Current Progress: " + str(self.pb['value']) + "%"
            


class GUI():
    def __init__(self):
        """Create GUI"""
        root = tk.Tk()
        root.title("HYPAT Dashboard")
        
        fig = Figure(figsize = (5, 5), dpi = 100)
        self.plot1 = fig.add_subplot(111)
        self.plot1.set_title("Loss Per Epochs")
        self.plot1.set_xlabel("Epochs")
        self.plot1.set_ylabel("Loss")
        self.canvas = FigureCanvasTkAgg(fig, master = root)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(column=2, row=0, rowspan=5)

        self.caption = ttk.Label(root, text="Epochs", font=("Arial", 15))
        self.caption.grid(column=0, row=0, sticky=tk.N, columnspan=2, padx=10, pady=(15,10))

        self.pb = ttk.Progressbar(root, orient='horizontal', mode='determinate', length=300)
        self.pb.grid(column=0, row=0, columnspan=2, padx=10)

        self.value_label = ttk.Label(root, text="Current Progress: 0%", font=("Arial", 11))
        self.value_label.grid(column=0, row=0, sticky=tk.S, columnspan=2, padx=10, pady=(0, 20))

        self.caption2 = ttk.Label(root, text="Batches", font=("Arial", 15))
        self.caption2.grid(column=0, row=1, sticky=tk.N, columnspan=2, padx=10, pady=(15,10))

        self.pb2 = ttk.Progressbar(root, orient='horizontal', mode='determinate', length=300)
        self.pb2.grid(column=0, row=1, columnspan=2, padx=10)

        self.value_label2 = ttk.Label(root, text="Current Progress: 0%", font=("Arial", 11))
        self.value_label2.grid(column=0, row=1, sticky=tk.S, columnspan=2, padx=10, pady=(0, 20))

        button = tk.Button(
            root, text="Start Training", command = self.threading_func,
            width=20, height=5, font=("Arial", 15)
        )
        button.grid(column=0, row=3, padx=(30, 0))

        root.mainloop()
    
    def threading_func(self):
        t1 = threading.Thread(target=self.work)
        t1.start()

    def work(self):
        #Initialize Model
        modeltrainer = ModelTrainer(self.plot1, self.canvas, self.pb, self.value_label, self.pb2, self.value_label2)
        modeltrainer.import_data()
        modeltrainer.main_training_loop()
    

GUI()
