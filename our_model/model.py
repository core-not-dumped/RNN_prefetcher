import math
import keras
import numpy as np
import tensorflow as tf
import pandas as pd
import shutil
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from keras.datasets import imdb
from keras.preprocessing import sequence
from tqdm import tqdm
from keras.layers import Input, Dense, Embedding, LSTM, Dropout, GRU
from tensorflow.compat.v1.keras.backend import set_session

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
set_session(session)

# ^ example for N line prefetcher
seq_length = 4 # length of sequence for a training example
output_length = 2 # look output size
start_position = 1 # delta start
epoch = 50
BATCH_SIZE = 64
DENSE_HIDDEN_SIZE = 20

FIRST_OUTPUT_SIZE = 256
SECOND_OUTPUT_SIZE = 64
FINAL_HIDDEN_SIZE = 256

PREDICT_SIZE_POW = 7

partition = 1 # partition or not
if partition == 1:
    MAXLEN = 42 - PREDICT_SIZE_POW + 1
else:
    MAXLEN = 42
ADDRESS_PREDICT_SIZE = 1 << PREDICT_SIZE_POW
EMBEDDING_DIM = 100
RNN_UNITS = 256
cut_off = 0.1

padding = 0
num_before_address = 2

def address_to_binary(a):
    binary_int = list()
    for i in range(MAXLEN):
        binary_int.append(a%2)
        a = a//2

    return binary_int

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


class MLPrefetchModel(object):
    '''
    Abstract base class for your models. For HW-based approaches such as the
    NextLineModel below, you can directly add your prediction code. For ML
    models, you may want to use it as a wrapper, but alternative approaches
    are fine so long as the behavior described below is respected.
    '''

    @abstractmethod
    def load(self, path):
        '''
        Loads your model from the filepath path
        '''
        pass

    @abstractmethod
    def save(self, path):
        '''
        Saves your model to the filepath path
        '''
        pass

    @abstractmethod
    def train(self, data):
        '''
        Train your model here. No return value. The data parameter is in the
        same format as the load traces. Namely,
        Unique Instr Id, Cycle Count, Load Address, Instruction Pointer of the Load, LLC hit/miss
        '''
        pass

    @abstractmethod
    def generate(self, data):
        '''
        Generate your prefetches here. Remember to limit yourself to 2 prefetches
        for each instruction ID and to not look into the future :).

        The return format for this will be a list of tuples containing the
        unique instruction ID and the prefetch. For example,
        [
            (A, A1),
            (A, A2),
            (C, C1),
            ...
        ]

        where A, B, and C are the unique instruction IDs and A1, A2 and C1 are
        the prefetch addresses.
        '''
        pass

class MementoModel(MLPrefetchModel):

    mapping = {}
    scores = defaultdict(int)
    last_ip_access = defaultdict(list)
    delay = eval(os.environ.get('MEMENTO_DELAY', '5'))
    llc = CacheSimulator(16, 2048, 64)

    def load(self, path):
        # Load your pytorch / tensorflow model from the given filepath
        print('Loading ' + path + ' for NextLineModel')

    def save(self, path):
        # Save your model to a file
        print('Saving ' + path + ' for NextLineModel')

    def train(self, data):
        '''
        Train your model here using the data

        The data is the same format given in the load traces. Namely:
        Unique Instr Id, Cycle Count, Load Address, Instruction Pointer of the Load, LLC hit/miss
        '''
        print('Training NextLineModel')

        for (instr_id, cycle_count, load_addr, load_ip, llc_hit) in data:
            if len(self.last_ip_access[load_ip]) >= self.delay:
                key = load_ip, self.last_ip_access[load_ip][0]
                self.mapping[key] = load_addr
                self.scores[key] += 1
            self.last_ip_access[load_ip].append(load_addr)
            if len(self.last_ip_access[load_ip]) > self.delay:
                self.last_ip_access[load_ip].pop(0)

    def generate(self, data):
        '''
        Generate the prefetches for the prefetch file for ChampSim here

        As a reminder, no looking ahead in the data and no more than 2
        prefetches per unique instruction ID

        The return format for this function is a list of (instr_id, pf_addr)
        tuples as shown below
        '''
        prefetches = []
        for (instr_id, cycle_count, load_addr, load_ip, llc_hit) in data:
            # Prefetch the next two blocks
            key = load_ip, load_addr
            if key in self.mapping and self.scores[key] > 0:
                prefetches.append((instr_id, self.mapping[key]))

        return prefetches


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

        if len(delta_list) < seq_length + output_length + padding + start_position:
            continue

        for j in list(range(start_position, len(delta_list) - seq_length - padding - output_length + 1)):

            if(MAXLEN == 42):
                address_binary_input.append(address_to_binary((address_delta[i] << (PREDICT_SIZE_POW - 1)) + offset_list[j + seq_length - 1]))
            else:
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
                prefetches.append((data[i][0], (load_address[i]  + 1) << 6))
                prefetches.append((data[i][0], (load_address[i]  + 2) << 6))
                continue

            rnn_data = access[load][1][-seq_length:]

            if(MAXLEN == 42):
                address_binary_input = address_to_binary(load_address[i])
            else:
                address_binary_input = address_to_binary(load)

            input_eval = [tf.expand_dims(rnn_data, 0), tf.expand_dims(address_binary_input, 0)]

            self.model.reset_states()
            predictions = self.model(input_eval)

            predictions_np = predictions.numpy()
            predictions_np = np.squeeze(predictions_np,axis=0)
            predictions_np_index = np.argsort(predictions_np)

            high_prob_index = 0
            prefetches_address = []

            # prefetches two
            cnt = 0
            while cnt < 2:
            
                high_prob_index -= 1
            
                if (load_address[i] + predictions_np_index[high_prob_index]) in access[load][2]:
                    continue
            
                prefetches.append((data[i][0], (load_address[i]  + predictions_np_index[high_prob_index] - (1 << (PREDICT_SIZE_POW - 1)) + 1) << 6))
                prefetches_address.append(load_address[i] + predictions_np_index[high_prob_index])
                cnt += 1
            
            prefetches_address = prefetches_address + access[load][2][:num_before_address]
            access[load][2] = prefetches_address

            # prefetches using cutoff
            '''while True:
            
                high_prob_index -= 1
            
                if (load_address[i] + predictions_np_index[high_prob_index]) in access[load][2]:
                    continue
                
                if predictions_np[predictions_np_index[high_prob_index]] < cut_off:
                    break

                prefetches.append((data[i][0], (load_address[i] + predictions_np_index[high_prob_index] - (1 << (PREDICT_SIZE_POW - 1)) + 1) << 6))
                prefetches_address.append(load_address[i] + predictions_np_index[high_prob_index])
            
            prefetches_address = prefetches_address + access[load][2][:2]
            access[load][2] = prefetches_address'''


        return prefetches


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


class BOMemento(BestOffset):

    def __init__(self) -> None:
        super().__init__()
        self.memento = MementoModel()

    def train(self, data):
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
        memento_prefetch_requests = []
        percent = len(data) // 100

        bo_useful = defaultdict(set)
        memento_useful = defaultdict(set)

        train_memento_data = data[:len(data) // 2]
        self.memento.train(train_memento_data)

        for i, (instr_id, cycle_count, load_addr, load_ip, llc_hit) in enumerate(data[len(data) // 2:]):
            hit, prefetched = self.llc.load(load_addr, False)
            if hit and prefetched:
                bo_useful[prefetched].add(load_addr)
            memento_hit, memento_prefetched = self.memento.llc.load(load_addr, False)
            if memento_hit and memento_prefetched:
                memento_useful[memento_prefetched].add(load_addr)

            # handle arrived prefetch requests for bo
            while prefetch_requests and prefetch_requests[0][0] + self.memory_latency < cycle_count:
                fill_addr = prefetch_requests[0][1]
                h, p = self.llc.load(fill_addr, prefetch_requests[0][2])
                if not h:
                    if self.best_index == -1:
                        fill_line_addr = fill_addr >> 6
                        if self.best_index != -1:
                            offset = self.offsets[self.best_index]
                        else:
                            offset = 0
                        self.rr_add_immediate(fill_line_addr - offset)
                prefetch_requests.pop(0)

            # handle arrived prefetch requests for bo
            while prefetch_requests and prefetch_requests[0][0] + self.memory_latency < cycle_count:
                fill_addr = prefetch_requests[0][1]
                h, p = self.llc.load(fill_addr, prefetch_requests[0][2])
                if not h:
                    if self.best_index == -1:
                        fill_line_addr = fill_addr >> 6
                        if self.best_index != -1:
                            offset = self.offsets[self.best_index]
                        else:
                            offset = 0
                        self.rr_add_immediate(fill_line_addr - offset)
                prefetch_requests.pop(0)

            # handle arrived memory accesses for memento
            while memento_prefetch_requests and memento_prefetch_requests[0][0] + self.memory_latency < cycle_count:
                fill_addr = memento_prefetch_requests[0][1]
                h, p = self.memento.llc.load(fill_addr, True)
                memento_prefetch_requests.pop(0)

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
                    prefetch_requests.append((cycle_count, addr_1, instr_id))
                    prefetch_requests.append((cycle_count, addr_2, instr_id))
            else:
                pass

            # prefetch for memento
            for iid, address in self.memento.generate([(instr_id, cycle_count, load_addr, load_ip, llc_hit)]):
                memento_prefetch_requests.append((cycle_count, address, instr_id))

            if i % percent == 0:
                print(i // percent, self.active_offsets, self.best_index_score,
                      sum(self.acc) / 2 / (len(self.acc) + 1),
                      sum(self.acc_alt) / 2 / (len(self.acc_alt) + 1))
                print('useful bo', len(bo_useful), 'memento', len(memento_useful))
                self.acc.clear()
                self.acc_alt.clear()
                self.active_offsets.clear()

        history = 4
        classification_data = []
        classification_labels = []
        offset_history = []
        for i, (instr_id, cycle_count, load_addr, load_ip, llc_hit) in enumerate(data[len(data) // 2:]):
            offset_history.append(load_addr)
            if len(offset_history) > history + 1:
                offset_history.pop(0)
            else:
                continue
            classification_data.append([offset_history[i] - offset_history[i - 1] for i in range(1, len(offset_history))])
            if len(bo_useful[instr_id]) >= len(memento_useful[instr_id]):
                classification_labels.append(1)
            else:
                classification_labels.append(0)

        return prefetches


class MultiSaturatingCounter:

    def __init__(self, keys, limit=64) -> None:
        super().__init__()
        self.counters = {key: 0 for key in keys}
        self.limit = limit

    def promote(self, key, factor=1):
        self.counters[key] += factor * len(self.counters)
        for key in self.counters:
            self.counters[key] -= factor
        for key in self.counters:
            if self.counters[key] < -self.limit:
                self.counters[key] = -self.limit
            if self.counters[key] > self.limit:
                self.counters[key] = self.limit

    @property
    def best_order(self):
        return [-p for _, p in sorted([(v, -k) for k, v in self.counters.items()], reverse=True)]


class SetDueler(MLPrefetchModel):

    prefetcher_classes = (BestOffset,
                          rp4_Model, )

    demotion_factor = eval(os.environ.get('DUELER_DEMOTION_FACTOR', '1'))
    promotion_factor = eval(os.environ.get('DUELER_PROMOTION_FACTOR', '0'))

    def __init__(self) -> None:
        super().__init__()
        self.prefetchers = [prefetcher_class() for prefetcher_class in self.prefetcher_classes]

    def load(self, path):
        for prefetcher in self.prefetchers:
            prefetcher.load(path)

    def save(self, path):
        for prefetcher in self.prefetchers:
            prefetcher.save(path)

    def train(self, data):
        # data = data[:len(data) // 5]
        for prefetcher in self.prefetchers:
            prefetcher.train(data)

    def generate(self, data):
        # data = data[:len(data) // 50]
        prefetch_sets = defaultdict(lambda: defaultdict(list))
        memory_latency = 200
        for p, prefetcher in enumerate(self.prefetchers):
            prefetches = prefetcher.generate(data)
            for iid, addr in prefetches:
                prefetch_sets[p][iid].append((iid, addr))
        total_prefetches = []
        prefetch_request_sets = {p: [] for p, _ in enumerate(self.prefetchers)}
        counters = MultiSaturatingCounter(range(len(self.prefetchers)))
        cache_models = []

        def demotion(cache_name, address, tag, label):
            if label:
                counters.promote(p, self.demotion_factor)
        for p, _ in enumerate(self.prefetchers):
            cache_models.append(CacheSimulator(16, 2048, 64, eviction_hook=demotion, name=p))
        for i, (instr_id, cycle_count, load_addr, load_ip, llc_hit) in enumerate(data):

            for p, _ in enumerate(self.prefetchers):
                while prefetch_request_sets[p] and prefetch_request_sets[p][0][0] + memory_latency < cycle_count:
                    h, l = cache_models[p].load(prefetch_request_sets[p][0][1], True)
                    prefetch_request_sets[p].pop(0)

            for p, _ in enumerate(self.prefetchers):
                hit, prefetched = cache_models[p].load(load_addr, False, overwrite=True)
                if prefetched:
                    counters.promote(p, self.promotion_factor)
            instr_prefetches = []
            for winner in counters.best_order:
                instr_prefetches.extend(prefetch_sets[winner][instr_id])
                break
            for winner in counters.best_order:
                for iid, paddr in prefetch_sets[winner][instr_id]:
                    prefetch_request_sets[winner].append((cycle_count, paddr))
            instr_prefetches = instr_prefetches[:2]
            total_prefetches.extend(instr_prefetches)
        return total_prefetches


class Hybrid(MLPrefetchModel):

    prefetcher_classes = (rp4_Model,
                          BestOffset, )

    def __init__(self) -> None:
        super().__init__()
        self.prefetchers = [prefetcher_class() for prefetcher_class in self.prefetcher_classes]

    def load(self, path):
        pass

    def save(self, path):
        pass

    def train(self, data):
        for prefetcher in self.prefetchers:
            prefetcher.train(data)

    def generate(self, data):
        prefetch_sets = defaultdict(lambda: defaultdict(list))
        for p, prefetcher in enumerate(self.prefetchers):
            prefetches = prefetcher.generate(data)
            for iid, addr in prefetches:
                prefetch_sets[p][iid].append((iid, addr))
        total_prefetches = []

        for i, (instr_id, cycle_count, load_addr, load_ip, llc_hit) in enumerate(data):

            instr_prefetches = []
            for d in range(2):
                for p in range(len(self.prefetchers)):
                    if prefetch_sets[p][instr_id]:
                        instr_prefetches.append(prefetch_sets[p][instr_id].pop(0))
            if(instr_prefetches[0][1] == instr_prefetches[1][1]):
                instr_prefetches = instr_prefetches.append(instr_prefetches[0])
                instr_prefetches = instr_prefetches.append(instr_prefetches[2])
            else:
                instr_prefetches = instr_prefetches[:2]
            total_prefetches.extend(instr_prefetches)
        return total_prefetches

class perfect_model(MLPrefetchModel):
    def load(self, path):
        print('load')

    def save(self, path):
        print('save')

    def train(self, data):
        print('train')

    def generate(self, data):
        prefetches = []

        pre_number = 5

        for i in range(len(data) - pre_number - 2):
            prefetches.append((data[i][0], data[i+pre_number][2]))
            prefetches.append((data[i][0], data[i+pre_number+1][2]))
            prefetches.append((data[i][0], data[i+pre_number+2][2]))
        
        #for i in range(len(data)):
        #    
        #    for j in range(i+1, len(data) - 1):
        #        if data[j][1] - data[i][1] > 5000:
        #            prefetches.append((data[i][0], data[j][2]))
        #            prefetches.append((data[i][0], data[j+1][2]))
        #            break

        return prefetches



# ml_model_name = os.environ.get('ML_MODEL_NAME', 'Hybrid')
# Model = eval(ml_model_name)

Model = rp4_Model
