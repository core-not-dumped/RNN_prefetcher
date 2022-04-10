import keras
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import shutil
from abc import ABC, abstractmethod
from keras.datasets import imdb
from keras.preprocessing import sequence
from tqdm import tqdm
from keras.layers import Input, Dense, Embedding, LSTM, Dropout, GRU
from tensorflow.compat.v1.keras.backend import set_session

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
set_session(session)

class MLPrefetchModel(object):
    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def generate(self, data):
        pass

class NextLineModel(MLPrefetchModel):

    def load(self, path):
        print('Loading ' + path + ' for NextLineModel')

    def save(self, path):
        print('Saving ' + path + ' for NextLineModel')

    def train(self, data):
        print('Training NextLineModel')

    def generate(self, data):
        print('Generating for NextLineModel')
        prefetches = []
        for (instr_id, cycle_count, load_addr, load_ip, llc_hit) in data:
            # Prefetch the next two blocks
            prefetches.append((instr_id, ((load_addr >> 6) + 1) << 6))
            prefetches.append((instr_id, ((load_addr >> 6) + 2) << 6))

        return prefetches

# ^ example for N line prefetcher
seq_length = 4 # length of sequence for a training example
output_length = 1
epoch = 100
BATCH_SIZE = 64
MAXLEN = 33
DENSE_HIDDEN_SIZE = 20

FIRST_OUTPUT_SIZE = 256
SECOND_OUTPUT_SIZE = 32
FINAL_HIDDEN_SIZE = 256

PREDICT_SIZE_POW = 7
ADDRESS_PREDICT_SIZE = 1 << PREDICT_SIZE_POW
EMBEDDING_DIM = 256
RNN_UNITS = 1024
cut_off = 0.3

text_file = '473.astar-s0'

def address_to_binary(a):
    binary_int = list()
    for i in range(MAXLEN):
        # /1000 -> scaling
        binary_int.append(a%2)
        a = a//2
    return binary_int

#################################################################################
################################## model 1 ######################################
#################################################################################

def build_rp1(address_predict_size, embedding_dim, rnn_units):
    input_x = Input(shape=(seq_length,))
    x = Embedding(address_predict_size, embedding_dim)(input_x)
    x = LSTM(rnn_units, recurrent_initializer='glorot_uniform') (x)
    x = Dense(address_predict_size, activation="sigmoid") (x)
    model = keras.models.Model(inputs=input_x, outputs=x)
    return model

def make_train_set(load_address, generate = False):
    '''make delta'''
    delta = list()
    for i in range(len(load_address)-1):
        delta.append(load_address[i+1] - load_address[i])

    for i in range(len(delta)):
        if delta[i] > 500:      delta[i] = 1000
        elif delta[i] < -500:   delta[i] = 0
        else:                   delta[i] += 500

    delta_bundle = list()
    for i in range(len(delta) - seq_length):
        delta_bundle.append(delta[i:i+seq_length+1])
    delta_bundle = np.array(delta_bundle)

    '''make address_binary_input'''
    address_binary_input = list()
    for i in range(seq_length, len(load_address)-1):
        address_binary_input.append(address_to_binary(load_address[i]))

    '''intput output data'''
    rnn_data = delta_bundle[:,:-1]
    rnn_data = np.array(rnn_data)

    delta_output = delta_bundle[:,-1:]
    delta_output = delta_output.flatten()

    address_binary_input = np.array(address_binary_input)

    if generate == False:
        '''delete garbage data'''
        delete_list = list()
        for i in range(len(delta_output)):
            if delta_output[i] == 0 or delta_output[i] == 1000:
                delete_list.append(i)

        print(f"delete_list_len : {len(delete_list)}")

        delta_output = np.delete(delta_output, delete_list)
        rnn_data = np.delete(rnn_data, delete_list, 0)
        address_binary_input = np.delete(address_binary_input, delete_list, 0)

        '''shuffle data'''
        idx = np.arange(rnn_data.shape[0])
        np.random.shuffle(idx)

        rnn_data = rnn_data[idx]
        delta_output = delta_output[idx]
        address_binary_input = address_binary_input[idx]

    return rnn_data, address_binary_input, delta_output


# rnn_prefetcher_1 model
class rp1_Model(MLPrefetchModel):
    def __init__(self):
        self.model = build_rp1(ADDRESS_PREDICT_SIZE, EMBEDDING_DIM, RNN_UNITS)
    
    def load(self, path):
        self.model = tf.keras.models.load_model(path)
    
    def save(self, path):
        self.model.save(path)

    def train(self, data):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        load_address = list()
        for (instr_id, cycle_count, load_addr, load_ip, llc_hit) in data:
            load_address.append(load_addr >> 6)

        (rnn_data, address_binary_input, delta_output) = make_train_set(load_address)

        history = self.model.fit(rnn_data, delta_output, batch_size=BATCH_SIZE, epochs=epoch)

    def generate(self, data):
        prefetches = []

        load_address = list()
        for (instr_id, cycle_count, load_addr, load_ip, llc_hit) in data:
            load_address.append(load_addr >> 6)

        (rnn_data, delta_output) = make_train_set(load_address, generate = True)

        for i in tqdm(range(len(rnn_data))):
            input_eval = tf.expand_dims(rnn_data[i], 0)

            self.model.reset_states()
            predictions = self.model(input_eval)

            predictions_np = predictions.numpy()
            predictions_np = np.squeeze(predictions_np,axis=0)
            predictions_np = np.argsort(predictions_np)

            prefetches.append((data[seq_length+i][0], ((data[seq_length+i][2] >> 6) + (predictions_np[-1] - 500)) << 6))
            prefetches.append((data[seq_length+i][0], ((data[seq_length+i][2] >> 6) + (predictions_np[-2] - 500)) << 6))

        return prefetches

#################################################################################
################################## model 2 ######################################
#################################################################################

def build_rp2(address_predict_size, first_output_size, second_output_size, final_hidden_size, embedding_dim, rnn_units):
  
    input_x = Input(shape=(seq_length,))
    x = Embedding(address_predict_size, embedding_dim)(input_x)
    x = LSTM(rnn_units, recurrent_initializer='glorot_uniform') (x)
    x = Dense(first_output_size, activation="sigmoid") (x)
    x = keras.models.Model(inputs=input_x, outputs=x)

    input_y = Input(shape=(MAXLEN,))
    y = Dense(second_output_size, activation="relu")(input_y)
    y = keras.models.Model(inputs=input_y, outputs=y)

    combined = tf.keras.layers.concatenate([x.output, y.output])

    z = Dense(final_hidden_size, activation="relu")(combined)
    z = Dense(address_predict_size, activation="sigmoid")(z)

    model = keras.models.Model(inputs=[x.input, y.input], outputs=z)

    return model

# rnn_prefetcher_2 model
class rp2_Model(MLPrefetchModel):
    def __init__(self):
        self.model = build_rp2(ADDRESS_PREDICT_SIZE, FIRST_OUTPUT_SIZE, SECOND_OUTPUT_SIZE, FINAL_HIDDEN_SIZE, EMBEDDING_DIM, RNN_UNITS)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)

    def save(self, path):
        self.model.save(path)

    def train(self, data):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        load_address = list()
        for (instr_id, cycle_count, load_addr, load_ip, llc_hit) in data:
            load_address.append(load_addr >> 6)

        (rnn_data, address_binary_input, delta_output) = make_train_set(load_address)

        history = self.model.fit([rnn_data, address_binary_input], delta_output, batch_size=BATCH_SIZE, epochs=epoch)

    def generate(self, data):
        prefetches = []

        load_address = list()
        for (instr_id, cycle_count, load_addr, load_ip, llc_hit) in data:
            load_address.append(load_addr >> 6)

        (rnn_data, address_binary_input, delta_output) = make_train_set(load_address, generate = True)

        for i in tqdm(range(len(rnn_data))):

            num_generate = 2

            input_eval = [tf.expand_dims(rnn_data[i].tolist(), 0), tf.expand_dims(address_binary_input[i].tolist(), 0)]

            self.model.reset_states()
            predictions = self.model(input_eval)

            predictions_np = predictions.numpy()
            predictions_np = np.squeeze(predictions_np,axis=0)
            predictions_np = np.argsort(predictions_np)

            prefetches.append((data[seq_length+i][0], ((data[seq_length+i][2] >> 6) + (predictions_np[-1] - 500)) << 6))
            prefetches.append((data[seq_length+i][0], ((data[seq_length+i][2] >> 6) + (predictions_np[-2] - 500)) << 6))

        return prefetches

#################################################################################
################################## model 3 ######################################
#################################################################################

def build_rp3(address_predict_size, first_output_size, second_output_size, final_hidden_size, embedding_dim, rnn_units):
  
    input_x = Input(shape=(seq_length,))
    x = Embedding(address_predict_size, embedding_dim)(input_x)
    x = LSTM(rnn_units, recurrent_initializer='glorot_uniform') (x)
    x = Dense(first_output_size, activation="sigmoid") (x)
    x = keras.models.Model(inputs=input_x, outputs=x)

    input_y = Input(shape=(MAXLEN,))
    y = Dense(second_output_size, activation="relu")(input_y)
    y = keras.models.Model(inputs=input_y, outputs=y)

    combined = tf.keras.layers.concatenate([x.output, y.output])

    z = Dense(final_hidden_size, activation="relu")(combined)
    z = Dropout(0.2, noise_shape=None, seed=None)(z)
    z = Dense(address_predict_size, activation="sigmoid")(z)

    model = keras.models.Model(inputs=[x.input, y.input], outputs=z)

    return model

def make_train_data_rp3(load_address):

    scale = 0.2
    weight_lean = 0.5
    address_binary_input = list()
    address = list()
    rnn_input = list()
    multi_output = list()
    categorical_multi_output = list()
    offset_dic = dict()

    '''make offset dictionary'''
    for i in range(len(load_address)):
        load = load_address[i] >> 10
        if load in offset_dic:      offset_dic[load].append(load_address[i] % (1 << 10))
        else:                       offset_dic[load] = [load_address[i] % (1 << 10)]

    offset_address = np.array(list(offset_dic.keys()))

    print('make_data_start')

    '''make data'''
    for i in tqdm(range(len(offset_address))):

        offset_list = offset_dic[offset_address[i]]
        if len(offset_list) <= seq_length:     continue

        for j in [0] + list(range(1, len(offset_list) - seq_length - output_length + 1)):
            address_binary_input.append(address_to_binary(offset_address[i]))
            rnn_input.append(offset_list[j:j+seq_length])
            categorical_multi_output.append(offset_list[j+seq_length])

            '''
            if len(offset_list) < seq_length + output_length:       multi_output = offset_list[j+seq_length:]
            else:                                                   multi_output = offset_list[j+seq_length:j+seq_length+output_length]

            tmp = [0 for k in range(1024)]
            for k in range(len(multi_output)):
                tmp[multi_output[k]] = round(1 - weight_lean * k/output_length,2)
                categorical_multi_output.append(tmp)
            '''

    print('make_data_finish')

    address_binary_input = np.array(address_binary_input)
    rnn_input = np.array(rnn_input)
    categorical_multi_output = np.array(categorical_multi_output)

    address_binary_input = scale * address_binary_input

    # shuffle_data
    print('shuffle data')
    idx = np.arange(address_binary_input.shape[0])
    np.random.shuffle(idx)

    print(f'{address_binary_input.shape}')
    print(f'{rnn_input.shape}')
    print(f'{categorical_multi_output.shape}')

    address_binary_input = address_binary_input[idx]
    rnn_input = rnn_input[idx]
    categorical_multi_output = categorical_multi_output[idx]

    return (rnn_input, address_binary_input, categorical_multi_output)

# rnn_prefetcher_3 model
class rp3_Model(MLPrefetchModel):
    def __init__(self):
        print('make_model')
        self.model = build_rp3(ADDRESS_PREDICT_SIZE, FIRST_OUTPUT_SIZE, SECOND_OUTPUT_SIZE, FINAL_HIDDEN_SIZE, EMBEDDING_DIM, RNN_UNITS)
        print('finish!')

    def load(self, path):
        self.model = tf.keras.models.load_model(path)

    def save(self, path):
        self.model.save(path)

    def train(self, data):
        print('compile')
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        print('finish!')

        load_address = list()
        for (instr_id, cycle_count, load_addr, load_ip, llc_hit) in data:
            load_address.append(load_addr >> 6)

        (rnn_data, address_binary_input, output_label) = make_train_data_rp3(load_address)

        history = self.model.fit([rnn_data, address_binary_input], output_label, batch_size=BATCH_SIZE, epochs=epoch)

    def generate(self, data):
        prefetches = []

        load_address = list()
        for (instr_id, cycle_count, load_addr, load_ip, llc_hit) in data:
            load_address.append(load_addr >> 6)

        access = dict()

        for i in tqdm(range(len(load_address))):

            num_generate = 2

            load = load_address[i] >> 10
            if load in access:      access[load].append(load_address[i] % (1 << 10))
            else:                   access[load] = [load_address[i] % (1 << 10)]

            if len(access[load]) < seq_length:
                prefetches.append((data[i][0], ((data[i][2] >> 6) + 1) << 6))
                prefetches.append((data[i][0], ((data[i][2] >> 6) + 2) << 6))
                continue

            rnn_data = access[load][-seq_length:]
            address_binary_input = address_to_binary(load)

            input_eval = [tf.expand_dims(rnn_data, 0), tf.expand_dims(address_binary_input, 0)]

            self.model.reset_states()
            predictions = self.model(input_eval)

            predictions_np = predictions.numpy()
            predictions_np = np.squeeze(predictions_np,axis=0)
            predictions_np = np.argsort(predictions_np)

            prefetches.append((data[i][0], (((data[i][2] >> 16) << 10) + predictions_np[-1]) << 6))
            prefetches.append((data[i][0], (((data[i][2] >> 16) << 10) + predictions_np[-2]) << 6))

        return prefetches

#################################################################################
################################## model 4 ######################################
#################################################################################


def build_rp4(address_predict_size, first_output_size, second_output_size, final_hidden_size, embedding_dim, rnn_units):
  
    input_x = Input(shape=(seq_length,))
    x = Embedding(address_predict_size - 1, embedding_dim)(input_x)
    x = LSTM(rnn_units, recurrent_initializer='glorot_uniform') (x)
    x = Dense(first_output_size, activation="relu") (x)
    x = keras.models.Model(inputs=input_x, outputs=x)

    input_y = Input(shape=(MAXLEN,))
    y = Dense(second_output_size, activation="relu")(input_y)
    y = keras.models.Model(inputs=input_y, outputs=y)

    combined = tf.keras.layers.concatenate([x.output, y.output])

    z = Dense(final_hidden_size, activation="relu")(combined)
    z = Dropout(0.2, noise_shape=None, seed=None)(z)
    z = Dense(address_predict_size - 1, activation="softmax")(z)

    model = keras.models.Model(inputs=[x.input, y.input], outputs=z)

    return model

def make_train_data_rp4(load_address):

    scale = 0.2
    delta_dic = dict()
    address_binary_input = list()
    rnn_input = list()
    output = list()

    for i in range(len(load_address)):
        load = load_address[i] >> (PREDICT_SIZE_POW - 1)
        offset = load_address[i] % (1 << (PREDICT_SIZE_POW - 1))
        if load in delta_dic:
            delta_dic[load][1].append(offset - delta_dic[load][0][-1] + (1 << (PREDICT_SIZE_POW - 1)) - 1)
            delta_dic[load][0].append(offset)
        else:
            delta_dic[load] = [[offset], [0]]

    address_delta = np.array(list(delta_dic.keys()))

    for i in range(len(address_delta)):
        offset_list = delta_dic[address_delta[i]][0]
        delta_list = delta_dic[address_delta[i]][1]
        
        if len(delta_list) <= seq_length:
            continue
        
        for j in [0] + list(range(1, len(delta_list) - seq_length - output_length + 1)):
            address_binary_input.append(address_to_binary((address_delta[i] << (PREDICT_SIZE_POW - 1)) + offset_list[j + seq_length - 1]))
            rnn_input.append(delta_list[j:j+seq_length])
            output.append(delta_list[j+seq_length])

    address_binary_input = np.array(address_binary_input)
    rnn_input = np.array(rnn_input)
    output = np.array(output)

    address_binary_input = scale * address_binary_input

    print('shuffle data')
    idx = np.arange(address_binary_input.shape[0])
    np.random.shuffle(idx)

    print(f'{address_binary_input.shape}')
    print(f'{rnn_input.shape}')
    print(f'{output.shape}')

    address_binary_input = address_binary_input[idx]
    rnn_input = rnn_input[idx]
    output = output[idx]

    return (rnn_input, address_binary_input, output)
    
# rnn_prefetcher_4 model
class rp4_Model(MLPrefetchModel):
    def __init__(self):
        print('make_model')
        self.model = build_rp4(ADDRESS_PREDICT_SIZE, FIRST_OUTPUT_SIZE, SECOND_OUTPUT_SIZE, FINAL_HIDDEN_SIZE, EMBEDDING_DIM, RNN_UNITS)
        print('finish!')

    def load(self, path):
        self.model = tf.keras.models.load_model(path)

    def save(self, path):
        self.model.save(path)

    def train(self, data):
        print('compile')
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print('finish!')

        load_address = list()
        for (instr_id, cycle_count, load_addr, load_ip, llc_hit) in data:
            load_address.append(load_addr >> 6)

        (rnn_data, address_binary_input, output_label) = make_train_data_rp4(load_address)

        history = self.model.fit([rnn_data, address_binary_input], output_label, batch_size=BATCH_SIZE, epochs=epoch)

    def generate(self, data):
        prefetches = []

        load_address = list()
        for (instr_id, cycle_count, load_addr, load_ip, llc_hit) in data:
            load_address.append(load_addr >> 6)

        access = dict()

        for i in tqdm(range(len(load_address))):

            num_generate = 2

            load = load_address[i] >> (PREDICT_SIZE_POW - 1)
            offset = load_address[i] % (1 << (PREDICT_SIZE_POW - 1))
            if load in access:      
                access[load][1].append(offset - access[load][0] + (1 << (PREDICT_SIZE_POW - 1)) - 1)
                access[load][0] = offset
            else:                   
                access[load] = [offset, [0]]

            if len(access[load][1]) < seq_length:
                prefetches.append((data[i][0], ((data[i][2] >> 6) + 1) << 6))
                prefetches.append((data[i][0], ((data[i][2] >> 6) + 2) << 6))
                continue

            rnn_data = access[load][1][-seq_length:]
            address_binary_input = address_to_binary(load_address[i])

            input_eval = [tf.expand_dims(rnn_data, 0), tf.expand_dims(address_binary_input, 0)]

            self.model.reset_states()
            predictions = self.model(input_eval)

            predictions_np = predictions.numpy()
            predictions_np = np.squeeze(predictions_np,axis=0)
            predictions_np_index = np.argsort(predictions_np)

            if i % 1000 == 0:
                print(predictions_np)

            if predictions_np[predictions_np_index[-1]] > cut_off:
                prefetches.append((data[i][0], (load_address[i] + predictions_np_index[-1] - (1 << (PREDICT_SIZE_POW - 1)) + 1) << 6))

            if predictions_np[predictions_np_index[-2]] > cut_off:
                prefetches.append((data[i][0], (load_address[i] + predictions_np_index[-2] - (1 << (PREDICT_SIZE_POW - 1)) + 1) << 6))
                
        return prefetches


###################################################################################
############################perfect model##########################################
###################################################################################

class perfect_model(MLPrefetchModel):
    def load(self, path):
        print('load')

    def save(self, path):
        print('save')

    def train(self, data):
        print('train')

    def generate(self, data):
        prefetches = []

        for i in range(len(data)-1):
            prefetches.append((data[i][0], data[i+1][2]))

        return prefetches

# Replace this if you create your own model
Model = rp4_Model
