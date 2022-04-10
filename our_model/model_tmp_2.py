import math
from collections import defaultdict

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
output_length = 2
epoch = 50
BATCH_SIZE = 64
DENSE_HIDDEN_SIZE = 20

FIRST_OUTPUT_SIZE = 256
SECOND_OUTPUT_SIZE = 64
FINAL_HIDDEN_SIZE = 256

PREDICT_SIZE_POW = 7
MAXLEN = 36;
ADDRESS_PREDICT_SIZE = 1 << PREDICT_SIZE_POW
EMBEDDING_DIM = 100
RNN_UNITS = 256
cut_off = 0.3

padding = 0

def address_to_binary(a):
    binary_int = list()
    for i in range(MAXLEN):
        binary_int.append(a%2)
        a = a//2

    return binary_int



#################################################################################
################################## BO model 1 ###################################
#################################################################################

class CacheSimulator(object):

    def __init__(self, sets, ways, block_size, eviction_hook=None, name=None) -> None:
        super().__init__()
        self.ways = ways
        self.name = name
        self.way_shift = int(math.log2(ways))
        self.sets = sets
        self.block_size = block_size
        self.block_shift = int(math.log2(block_size))
        self.storage = defaultdict(list)
        self.label_storage = defaultdict(list)
        self.eviction_hook = eviction_hook

    def parse_address(self, address):
        block = address >> self.block_shift
        way = block % self.ways
        tag = block >> self.way_shift
        return way, tag

    def load(self, address, label=None, overwrite=False):
        way, tag = self.parse_address(address)
        hit, l = self.check(address)
        if not hit:
            self.storage[way].append(tag)
            self.label_storage[way].append(label)
            if len(self.storage[way]) > self.sets:
                evicted_tag = self.storage[way].pop(0)
                evicted_label = self.label_storage[way].pop(0)
                if self.eviction_hook:
                    self.eviction_hook(self.name, address, evicted_tag, evicted_label)
        else:
            current_index = self.storage[way].index(tag)
            _t, _l = self.storage[way].pop(current_index), self.label_storage[way].pop(current_index)
            self.storage[way].append(_t)
            self.label_storage[way].append(_l)
        if overwrite:
            self.label_storage[way][self.storage[way].index(tag)] = label
        return hit, l

    def check(self, address):
        way, tag = self.parse_address(address)
        if tag in self.storage[way]:
            return True, self.label_storage[way][self.storage[way].index(tag)]
        else:
            return False, None

class BestOffset(MLPrefetchModel):
    offsets = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8, 9, -9, 10, -10, 11, -11, 12, -12, 13, -13, 14,
               -14, 15, -15, 16, -16, 18, -18, 20, -20, 24, -24, 30, -30, 32, -32, 36, -36, 40, -40]
    scores = [0 for _ in range(len(offsets))]
    round = 0
    best_index = 0
    second_best_index = 0
    best_index_score = 0
    temp_best_index = 0
    score_scale = eval(os.environ.get('BO_SCORE_SCALE', '1'))
    bad_score = int(10 * score_scale)
    low_score = int(20 * score_scale)
    max_score = int(31 * score_scale)
    max_round = int(100 * score_scale)
    llc = CacheSimulator(16, 2048, 64)
    rrl = {}
    rrr = {}
    dq = []
    acc = []
    acc_alt = []
    active_offsets = set()
    p = 0
    memory_latency = 200
    rr_latency = 60
    fuzzy = eval(os.environ.get('FUZZY_BO', 'False'))

    def load(self, path):
        # Load your pytorch / tensorflow model from the given filepath
        print('Loading ' + path + ' for BestOffset')

    def save(self, path):
        # Save your model to a file
        print('Saving ' + path + ' for BestOffset')

    def train(self, data):
        '''
        Train your model here using the data

        The data is the same format given in the load traces. Namely:
        Unique Instr Id, Cycle Count, Load Address, Instruction Pointer of the Load, LLC hit/miss
        '''
        print('Training BestOffset')

    def rr_hash(self, address):
        return ((address >> 6) + address) % 64

    def rr_add(self, cycles, address):
        self.dq.append((cycles, address))

    def rr_add_immediate(self, address, side='l'):
        if side == 'l':
            self.rrl[self.rr_hash(address)] = address
        elif side == 'r':
            self.rrr[self.rr_hash(address)] = address
        else:
            assert False

    def rr_pop(self, current_cycles):
        while self.dq:
            cycles, address = self.dq[0]
            if cycles < current_cycles - self.rr_latency:
                self.rr_add_immediate(address, side='r')
                self.dq.pop(0)
            else:
                break

    def rr_hit(self, address):
        return self.rrr.get(self.rr_hash(address)) == address or self.rrl.get(self.rr_hash(address)) == address

    def reset_bo(self):
        self.temp_best_index = -1
        self.scores = [0 for _ in range(len(self.offsets))]
        self.p = 0
        self.round = 0
        # self.acc.clear()
        # self.acc_alt.clear()

    def train_bo(self, address):
        testoffset = self.offsets[self.p]
        testlineaddr = address - testoffset

        if address >> 6 == testlineaddr >> 6 and self.rr_hit(testlineaddr):
            self.scores[self.p] += 1
            if self.scores[self.p] >= self.scores[self.temp_best_index]:
                self.temp_best_index = self.p

        if self.p == len(self.scores) - 1:
            self.round += 1
            if self.scores[self.temp_best_index] == self.max_score or self.round == self.max_round:
                self.best_index = self.temp_best_index if self.temp_best_index != -1 else 1
                self.second_best_index = sorted([(s, i) for i, s in enumerate(self.scores)])[-2][1]
                self.best_index_score = self.scores[self.best_index]
                if self.best_index_score <= self.bad_score:
                    self.best_index = -1
                self.active_offsets.add(self.best_index)
                self.reset_bo()
                return
        self.p += 1
        self.p %= len(self.scores)

    def generate(self, data):
        '''
        Generate the prefetches for the prefetch file for ChampSim here

        As a reminder, no looking ahead in the data and no more than 2
        prefetches per unique instruction ID

        The return format for this function is a list of (instr_id, pf_addr)
        tuples as shown below
        '''
        print('Generating for BestOffset')
        prefetches = []
        prefetch_requests = []
        percent = len(data) // 100
        for i, (instr_id, cycle_count, load_addr, load_ip, llc_hit) in enumerate(data):
            # Prefetch the next two blocks
            hit, prefetched = self.llc.load(load_addr, False)
            while prefetch_requests and prefetch_requests[0][0] + self.memory_latency < cycle_count:
                fill_addr = prefetch_requests[0][1]
                h, p = self.llc.load(fill_addr, True)
                if not h:
                    if self.best_index == -1:
                        fill_line_addr = fill_addr >> 6
                        if self.best_index != -1:
                            offset = self.offsets[self.best_index]
                        else:
                            offset = 0
                        self.rr_add_immediate(fill_line_addr - offset)
                prefetch_requests.pop(0)
            self.rr_pop(cycle_count)
            if not hit or prefetched:
                line_addr = (load_addr >> 6)
                self.train_bo(line_addr)
                self.rr_add(cycle_count, line_addr)
                if self.best_index != -1 and self.best_index_score > self.low_score:
                    addr_1 = (line_addr + 1 * self.offsets[self.best_index]) << 6
                    addr_2 = (line_addr + 2 * self.offsets[self.best_index]) << 6
                    addr_2_alt = (line_addr + 1 * self.offsets[self.second_best_index]) << 6
                    acc = len({addr_2 >> 6, addr_1 >> 6} & set(d[2] >> 6 for d in data[i + 1: i + 25]))
                    self.acc.append(acc)
                    acc_alt = len({addr_2_alt >> 6, addr_1 >> 6} & set(d[2] >> 6 for d in data[i + 1: i + 25]))
                    self.acc_alt.append(acc_alt)
                    # if acc_alt > acc:
                    #     addr_2 = addr_2_alt
                    prefetches.append((instr_id, addr_1))
                    prefetches.append((instr_id, addr_2))
                    prefetch_requests.append((cycle_count, addr_1))
                    prefetch_requests.append((cycle_count, addr_2))
            else:
                pass
            if i % percent == 0:
                print(i // percent, self.active_offsets, self.best_index_score,
                      sum(self.acc) / 2 / (len(self.acc) + 1),
                      sum(self.acc_alt) / 2 / (len(self.acc_alt) + 1))
                self.acc.clear()
                self.acc_alt.clear()
                self.active_offsets.clear()
        return prefetches

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
    z = Dense(address_predict_size - 1, activation="sigmoid")(z)

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
        
        if len(delta_list) < seq_length + output_length + padding + 1:
            continue
        
        for j in list(range(1, len(delta_list) - seq_length - padding - output_length + 1)):

            # binary_all
            #address_binary_input.append(address_to_binary((address_delta[i] << (PREDICT_SIZE_POW - 1)) + offset_list[j + seq_length - 1]))
            # binary_partion
            address_binary_input.append(address_to_binary(address_delta[i]))
            rnn_input.append(delta_list[j:j+seq_length])

            output_tmp = np.zeros(ADDRESS_PREDICT_SIZE-1)
            for k in range(output_length):
                output_tmp[delta_list[j + padding + seq_length + k]] = 1
            output.append(output_tmp)

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
        self.model.compile(optimizer='adam', loss='binary_crossentropy')
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

            # generate prefetches per request instructions
            num_before_address = 2
            num_generate = 2

            load = load_address[i] >> (PREDICT_SIZE_POW - 1)
            offset = load_address[i] % (1 << (PREDICT_SIZE_POW - 1))
            if load in access:
                access[load][1].append(offset - access[load][0] + (1 << (PREDICT_SIZE_POW - 1)) - 1)
                access[load][0] = offset
            else:                   
                access[load] = [offset, [0], [0 for i in range(num_before_address)]]

            # if there's not enough data
            if len(access[load][1]) < seq_length:
                continue

            rnn_data = access[load][1][-seq_length:]
            address_binary_input = address_to_binary(load)

            input_eval = [tf.expand_dims(rnn_data, 0), tf.expand_dims(address_binary_input, 0)]

            self.model.reset_states()
            predictions = self.model(input_eval)

            predictions_np = predictions.numpy()
            predictions_np = np.squeeze(predictions_np,axis=0)
            predictions_np_index = np.argsort(predictions_np)

            high_prob_index = 0
            prefetches_address = []

            cnt = 0
            while cnt < 2:

                high_prob_index -= 1

                if (load_address[i] + predictions_np_index[high_prob_index]) in access[load][2]:
                    continue

                prefetches.append((data[i][0], (load_address[i]  + predictions_np_index[high_prob_index] - (1 << (PREDICT_SIZE_POW - 1)) + 1) << 6))
                prefetches_address.append(load_address[i] + predictions_np_index[high_prob_index])
                cnt += 1

            prefetches_address = prefetches_address + access[load][2][:2]
            access[load][2] = prefetches_address
                
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

        pre_number = 2000

        for i in range(len(data) - pre_number):
            prefetches.append((data[i][0], data[i+pre_number][2]))
        
        #for i in range(len(data)):
        #    
        #    for j in range(i+1, len(data) - 1):
        #        if data[j][1] - data[i][1] > 5000:
        #            prefetches.append((data[i][0], data[j][2]))
        #            prefetches.append((data[i][0], data[j+1][2]))
        #            break

        return prefetches

# Replace this if you create your own model
Model = rp4_Model
