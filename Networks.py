"""
Created Monday 6/6/2022 12:56 PM

SPDX-FileCopyrightText: © 2022 Hudson Liu <hudsonliu0@gmail.com>

obama
"""

import keras
import random
import numpy as np

batch_size = 1 #only one batch since there's no gradient descent so batch size doesn't really matter
nlp_features = 97 #97 features because one hot encoding
nlp_embedding = 20 #just a random constant i decided on
nlp_nodes = 50
nlp_dense = 50
anet_nodes = 50
anet_pred_vars = 5 #All the parameters that anet can change
anet_dense = 50 #just for some extra complexity
hnet_pred_vars = 10 #same as above
hnet_dense = 50
hnet_first_layer = 50

nlp = keras.Sequential(name="Natural Langauge Processor")
nlp.add(keras.layers.LSTM(units = nlp_nodes, batch_input_shape = (batch_size, None, nlp_features))) #when 1 it is 4 when 4 it is 16
nlp.add(keras.layers.Dense(units = nlp_dense, activation = "relu"))
nlp.add(keras.layers.Dense(units = nlp_embedding, activation = "relu"))
oobama = nlp.get_weights()

anet = keras.Sequential(name="Architecture Network")
anet.add(keras.layers.LSTM(units = anet_nodes, activation = "relu", stateful = True, batch_input_shape = (batch_size, None, nlp_embedding)))
anet.add(keras.layers.Dense(units = anet_dense))
anet.add(keras.layers.Dense(units = anet_pred_vars))
obama = anet.get_weights()

hnet = keras.Sequential(name="Hyperparameter Network")
hnet.add(keras.layers.Dense(units = hnet_first_layer, input_shape=(nlp_embedding,)))
hnet.add(keras.layers.Dense(units = hnet_dense))
hnet.add(keras.layers.Dense(units = hnet_pred_vars))

#Preprocess the description and convert it into something the NLP can process (This step will be done before the gen alg is ran)
file = open("descriptions.txt", "r", encoding='utf-8-sig')
descriptions = file.readlines()
nl_removed = [description[:-2] for description in descriptions[:-1]] #remove the \n's for all of the lines; nl_removed means new lines removed
nl_removed.append(descriptions[-1])

#one hot encoding
one_hot_encoded = []
for data in nl_removed:
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
    char_to_int = dict((character, index) for index, character in enumerate(alphabet)) #give each character an index and then converting to dict

    integer_encoded = []
    for char in data:
        try:
            integer_encoded.append(char_to_int[char])
        except:
            continue

    desc_encoded = []
    for value in integer_encoded:
        letter = [0 for i in range(len(alphabet))]
        letter[value] = 1
        desc_encoded.append(letter)
    one_hot_encoded.append(desc_encoded)

#generate random weights and biases for LSTM layer
def rand_lstm_weights(input_shape, lstm_nodes, n_dense):
    wb = [] #wb is for weights and biases

    #randomly set values of lstm nodes
    U = [[(20 * random.random() - 10) for i in range(0, lstm_nodes * 4)] for i in range(0, lstm_nodes)] #weights can be any value but we'll restrict it between -10 and 10
    U1 = np.array(U, dtype=float)
    b = [(20 * random.random() - 10) for i in range(0, lstm_nodes * 4)] #bias can be any value
    b1 = np.array(b, dtype=float)
    W = [[(20 * random.random() - 10) for i in range(0, lstm_nodes * 4)] for i in range(0, input_shape)] #weights can be any value
    W1 = np.array(W, dtype=float)
    wb.append(W1)
    wb.append(U1)
    wb.append(b1)

    wb = rand_dense_weights(lstm_nodes, n_dense, wb)

    return wb

def rand_dense_weights(input_shape, n_dense, wb = []):
    #sets weights and biases
    i = 0
    for layer in n_dense:
        #randomly set weights, either referencing the previous layer or the lstm layer
        if i != 0:
            weights = np.array([[(20 * random.random() - 10) for i in range(0, layer)] for l in range(0, n_dense[i-1])], dtype=float)
        else: 
            weights = np.array([[(20 * random.random() - 10) for i in range(0, layer)] for l in range(0, input_shape)], dtype=float)
        wb.append(weights)

        #there is no input layer (lstm directly feeds to dense) and output layer has biases
        biases = np.array([(20 * random.random() - 10) for i in range(0, layer)], dtype=float)
        wb.append(biases)
        i += 1
    
    return wb

#tweaks the best individuals's weights and biases by up to 1
def tweaker(wb):
    for i in range(0, len(wb)): 
        if type(wb[i]) == float:
            if (random.random() < 0.25): #there is a 25% chance that it gets modified by up to plus-or-minus 1
                    wb[i] += (2.0 * random.random() - 1.0)
        else:
            wb[i] = tweaker(wb[i])
    return wb

#initialize the weights and biases of the nlp, anet, and hnet randomly
nlp_wb = rand_lstm_weights(input_shape = nlp_features, lstm_nodes = nlp_nodes, n_dense = [nlp_dense, nlp_embedding])
nlp.set_weights(nlp_wb)
anet_wb = rand_lstm_weights(input_shape = nlp_embedding, lstm_nodes = anet_nodes, n_dense = [anet_dense, anet_pred_vars])
anet.set_weights(anet_wb)
hnet_wb = rand_dense_weights(input_shape = nlp_embedding, n_dense = [hnet_first_layer, hnet_dense, hnet_pred_vars])
hnet.set_weights(hnet_wb)

#forward propagation


#anet timestep loop
