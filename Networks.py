"""
Created Monday 6/6/2022 12:56 PM

joebama
"""

import keras
import random
import numpy as np

batch_size = 1 #only one batch since there's no gradient descent so batch size doesn't really matter
nlp_features = 95 #95 features because one hot encoding
nlp_embedding = 20 #just a random constant i decided on
nlp_nodes = 50
nlp_dense = 50
anet_nodes = 50
#anet_pred_vars = 5 #All the parameters that anet can change
anet_dense = 50 #just for some extra complexity
hnet_pred_vars = 9 #same as anet pred vars
hnet_dense = 50
hnet_first_layer = 50

nlp = keras.Sequential(name="Natural_Langauge_Processor")
nlp.add(keras.layers.LSTM(units = nlp_nodes, batch_input_shape = (batch_size, None, nlp_features))) #when 1 it is 4 when 4 it is 16
nlp.add(keras.layers.Dense(units = nlp_dense, activation = "relu"))
nlp.add(keras.layers.Dense(units = nlp_embedding, activation = "sigmoid"))

anet_input_layer = keras.Input(shape = (None, nlp_embedding), batch_size=batch_size)
anet_lstm_layer = keras.layers.LSTM(units = anet_nodes, activation = "relu", stateful = True)(anet_input_layer)
anet_dense_layer = keras.layers.Dense(units = anet_dense, activation = "relu")(anet_lstm_layer)
relu_output = keras.layers.Dense(units = 22, activation = "relu")(anet_dense_layer)
sigmoid_output = keras.layers.Dense(units = 3, activation = "sigmoid")(anet_dense_layer) #dropout, padding, and stopping value
anet_output_layer = keras.layers.concatenate([relu_output, sigmoid_output])
anet = keras.Model(inputs=anet_input_layer, outputs=anet_output_layer, name="Architecture_Network")
obama = anet.get_weights()

hnet = keras.Sequential(name="Hyperparameter_Network")
hnet.add(keras.layers.Dense(units = hnet_first_layer, batch_input_shape=(batch_size, nlp_embedding,)))
hnet.add(keras.layers.Dense(units = hnet_dense, activation = "relu"))
hnet.add(keras.layers.Dense(units = hnet_pred_vars, activation = "sigmoid"))


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
def rand_lstm_weights(input_shape, lstm_nodes, n_dense, custom = False):
    wb = [] #wb is for weights and biases

    #randomly set values of lstm nodes
    U = [[(2 * random.random() - 1) for i in range(0, lstm_nodes * 4)] for i in range(0, lstm_nodes)] #weights can be any value but we'll restrict it between -10 and 10
    U1 = np.array(U, dtype=float)
    b = [(2 * random.random() - 1) for i in range(0, lstm_nodes * 4)] #bias can be any value
    b1 = np.array(b, dtype=float)
    W = [[(2 * random.random() - 1) for i in range(0, lstm_nodes * 4)] for i in range(0, input_shape)] #weights can be any value
    W1 = np.array(W, dtype=float)
    wb.append(W1)
    wb.append(U1)
    wb.append(b1)

    if not custom:
        wb = rand_dense_weights(lstm_nodes, n_dense, wb)
    else:
        wb = rand_dense_weights_custom(lstm_nodes, n_dense, wb)

    return wb

def rand_dense_weights(input_shape, n_dense, wb = []):
    #sets weights and biases
    i = 0
    for layer in n_dense:
        #randomly set weights, either referencing the previous layer or the lstm layer
        if i != 0:
            weights = np.array([[(2 * random.random() - 1) for i in range(0, layer)] for l in range(0, n_dense[i-1])], dtype=float)
        else: 
            weights = np.array([[(2 * random.random() - 1) for i in range(0, layer)] for l in range(0, input_shape)], dtype=float)
        wb.append(weights)

        #there is no input layer (lstm directly feeds to dense) and output layer has biases
        biases = np.array([(2 * random.random() - 1) for i in range(0, layer)], dtype=float)
        wb.append(biases)
        i += 1
    
    return wb

def rand_dense_weights_custom(input_shape, n_dense, wb = []):
    """Accounts for anet's structure specifically"""
    wb.append(np.array([[(2 * random.random() - 1) for i in range(0, n_dense[0])] for l in range(0, input_shape)], dtype=float)) 
    wb.append(np.array([(2 * random.random() - 1) for i in range(0, n_dense[0])], dtype=float))

    wb.append(np.array([[(2 * random.random() - 1) for i in range(0, n_dense[1])] for l in range(0, n_dense[0])], dtype=float)) 
    wb.append(np.array([(2 * random.random() - 1) for i in range(0, n_dense[1])], dtype=float))

    wb.append(np.array([[(2 * random.random() - 1) for i in range(0, n_dense[2])] for l in range(0, n_dense[0])], dtype=float)) 
    wb.append(np.array([(2 * random.random() - 1) for i in range(0, n_dense[2])], dtype=float))
    
    return wb

#tweaks the best individuals's weights and biases by up to 0.1
def mutation(wb):
    for i in range(0, len(wb)): 
        if type(wb[i]) == float:
            if (random.random() < 0.1): #there is a 10% chance that a weight or bias gets modified by up to plus-or-minus 0.1
                    wb[i] += (0.1 * random.random() - 0.05)
        else:
            wb[i] = mutation(wb[i])
    return wb

#initialize the weights and biases of the nlp, anet, and hnet randomly
nlp_wb = rand_lstm_weights(input_shape = nlp_features, lstm_nodes = nlp_nodes, n_dense = [nlp_dense, nlp_embedding])
nlp.set_weights(nlp_wb)
anet_wb = rand_lstm_weights(input_shape = nlp_embedding, lstm_nodes = anet_nodes, n_dense = [anet_dense, 22, 3], custom = True)
anet.set_weights(anet_wb)
hnet_wb = rand_dense_weights(input_shape = nlp_embedding, n_dense = [hnet_first_layer, hnet_dense, hnet_pred_vars])
hnet.set_weights(hnet_wb)

def interpret_output(hyperparameters, architecture):
    optimizer_output = hyperparameters[0, :9]

#def build_snet():

#forward propagation
for description in one_hot_encoded:
    embedding = nlp.predict([description])
    hyperparameters = hnet.predict(embedding)

    #forward propagate anet until it gets to a stopping point
    embedding_lstm = np.reshape(embedding, (1, 1, np.size(embedding)))
    architecture = [anet.predict(embedding_lstm)]
    i = 0
    while architecture[i][23] <= 0.5: #24 is reserved as a sigmoid stopping function
        architecture.append(anet.predict(embedding_lstm))
        i += 1
    anet.reset_states()

    interpret_output(hyperparameters, architecture)

#use generated parameters on test network


#calculate fitness based off of test network results

#purge the weak and strengthen the strong
