"""##auto"""

# Commented out IPython magic to ensure Python compatibility.
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Input, Dense, Embedding, LSTM
from keras.models import Model
import keras
import tensorflow as tf
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

seq_length = 5 # length of sequence for a training example
epoch = 30
MAXLEN = 43
DENSE_HIDDEN_SIZE = 20
train_model_percentage = 90

ADDRESS_PREDICT_SIZE = 1001
EMBEDDING_DIM = 64
RNN_UNITS = 1024

BUFFER_SIZE = 10000

f = open("./trace_file/473.astar-s0.txt", 'r')

load_address = list()

f.seek(0, 0)
while True:
  line = f.readline()
  if not line: break
  split_line = line.split(', ')
  load_address.append(split_line[2])

for i in range(len(load_address)):
  load_address[i] = int(load_address[i], 16)//64

print("\nload_address example: ")
print(load_address[:100])

delta = list()
for i in range(len(load_address)-1):
  delta.append(load_address[i+1] - load_address[i])

for i in range(len(delta)):
  if delta[i] > 500:
    delta[i] = 1000
    continue
  if delta[i] < -500:
    delta[i] = 0
    continue
  delta[i] += 500

print("\ndelta example, final delta:")
print(delta[0:100])
print(delta[-1])

delta_bundle = list()
for i in range(len(delta) - seq_length):
  delta_bundle.append(delta[i:i+seq_length+1])
delta_bundle = np.array(delta_bundle)

print("\ndelta_bundle example:")
print(delta_bundle)

rnn_data = delta_bundle[:,:-1]
rnn_data = np.array(rnn_data)
print("\nfirst input example(rnn_data): ")
print(rnn_data)

# 8 -> [0,0,0,1] change int to categorical value
def address_to_binary(a):
  binary_int = list()
  for i in range(MAXLEN):
    # /1000 -> scaling
    binary_int.append(a%2)
    a = a//2
  return binary_int

address_binary_input = list()
for i in range(seq_length, len(load_address)-1):
  address_binary_input.append(address_to_binary(load_address[i]))

print("\naddress binary input example")
print(address_binary_input[0])
address_binary_input = np.array(address_binary_input)

delta_output = delta_bundle[:,-1:]
delta_output = delta_output.flatten()
print("\noutput example:")
print(delta_output)

print('output shape')
print(rnn_data.shape)
print(address_binary_input.shape)
print(delta_output.shape)

delete_list = list()
for i in range(len(delta_output)):
  if delta_output[i] == 0 or delta_output[i] == 1000:
    delete_list.append(i)

delta_output = np.delete(delta_output, delete_list)
rnn_data = np.delete(rnn_data, delete_list, 0)
address_binary_input = np.delete(address_binary_input, delete_list, 0)

print("\n0 and 1000 removed delta output example:")
print(delta_output)

idx = np.arange(rnn_data.shape[0])
np.random.shuffle(idx)

rnn_data = rnn_data[idx]
address_binary_input = address_binary_input[idx]
delta_output = delta_output[idx]

cut_index = len(rnn_data) * train_model_percentage // 100

train_rnn_data = rnn_data[:cut_index,:]
test_rnn_data = rnn_data[cut_index:]

train_address_binary_input = address_binary_input[:cut_index,:]
test_address_binary_input = address_binary_input[cut_index:,:]

train_delta_output = delta_output[:cut_index]
test_delta_output = delta_output[cut_index:]

idx = np.arange(train_rnn_data.shape[0])
np.random.shuffle(idx)

train_rnn_data = train_rnn_data[idx]
train_address_binary_input = train_address_binary_input[idx]
train_delta_output = train_delta_output[idx]

idx = np.arange(test_rnn_data.shape[0])
np.random.shuffle(idx)

test_rnn_data = test_rnn_data[idx]
test_address_binary_input = test_address_binary_input[idx]
test_delta_output = test_delta_output[idx]

def build_model(address_predict_size, first_output_size, second_output_size, final_hidden_size, embedding_dim, rnn_units):
  
  input_x = Input(shape=(seq_length,))
  x = Embedding(address_predict_size, embedding_dim)(input_x)
  x = LSTM(rnn_units, recurrent_initializer='glorot_uniform') (x)
  x = Dense(first_output_size, activation="sigmoid") (x)
  x = Model(inputs=input_x, outputs=x)

  input_y = Input(shape=(MAXLEN,))
  y = Dense(second_output_size, activation="relu")(input_y)
  y = Model(inputs=input_y, outputs=y)

  combined = tf.keras.layers.concatenate([x.output, y.output])

  z = Dense(final_hidden_size, activation="relu")(combined)
  z = Dense(address_predict_size, activation="sigmoid")(z)

  model = Model(inputs=[x.input, y.input], outputs=z)

  return model

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits)

def generate_address(model, start_address):
  # Evaluation step (generating address using the learned model)

  # Number of characters to generate
  num_generate = 2

  input_eval = [tf.expand_dims(start_address[0], 0), tf.expand_dims(start_address[1], 0)]

  # Empty string to store our results
  generated = []

  # Low temperatures results in more predictable.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  
  predictions = model(input_eval)

  predictions = predictions / temperature

  # using probability
  '''
  generated = tf.random.categorical(predictions_np, num_samples=num_generate)[-1,0].numpy()  
  '''

  # using argmax
  predictions_np = predictions.numpy()
  predictions_np = np.squeeze(predictions_np,axis=0)
  predictions_np = np.argsort(predictions_np)
  generated = predictions_np [-num_generate:]

  return generated

FIRST_OUTPUT_SIZE_array = [64, 128, 256, 512]
SECOND_OUTPUT_SIZE_array = [8, 16, 32]
FINAL_HIDDEN_SIZE_array = [128, 256, 512]
BATCH_SIZE_array = [32, 64, 128]

data = dict()
continue_check = 0

if os.path.isfile(f'seq_length_{seq_length}.csv') == True:
  continue_check = 1
  before_data = pd.read_csv(f"seq_length_{seq_length}.csv", names=["first_output_size", "second_output_size", "final_hidden_size", "batch_size", "epoch_num", "train_accuracy", "accuracy_except_0_and_1000", "first_number_accuracy", "second_number_accuracy"])
  print(before_data)


for FIRST_OUTPUT_SIZE in FIRST_OUTPUT_SIZE_array:
  for SECOND_OUTPUT_SIZE in SECOND_OUTPUT_SIZE_array:
    for FINAL_HIDDEN_SIZE in FINAL_HIDDEN_SIZE_array:
      for BATCH_SIZE in BATCH_SIZE_array:
        
        if os.path.isdir('./training_checkpoints') == True:
            shutil.rmtree('./training_checkpoints')

        print("\n\nnew model start!!")
        print(f"{FIRST_OUTPUT_SIZE}, {SECOND_OUTPUT_SIZE}, {FINAL_HIDDEN_SIZE}, {BATCH_SIZE}\n")

        epoch_num_array = [10, 15, 20, 25, 30]

        if continue_check == 1:
          tmp = before_data[before_data["first_output_size"] == FIRST_OUTPUT_SIZE]
          tmp = tmp[tmp["second_output_size"] == SECOND_OUTPUT_SIZE]
          tmp = tmp[tmp["final_hidden_size"] == FINAL_HIDDEN_SIZE]
          tmp = tmp[tmp["batch_size"] == BATCH_SIZE]
          if tmp.shape[0] == len(epoch_num_array):
            continue
          else:
            continue_check = 0
            epoch_num_array = epoch_num_array[tmp.shape[0]:]

        model = build_model(ADDRESS_PREDICT_SIZE, FIRST_OUTPUT_SIZE, SECOND_OUTPUT_SIZE, FINAL_HIDDEN_SIZE, EMBEDDING_DIM, RNN_UNITS)
        
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
        
        checkpoint_dir = './training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

        checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
              filepath=checkpoint_prefix,
              save_weights_only=True)
        
        history = model.fit([train_rnn_data, train_address_binary_input], train_delta_output, batch_size=BATCH_SIZE, epochs=epoch, callbacks=[checkpoint_callback])

        model = build_model(ADDRESS_PREDICT_SIZE, FIRST_OUTPUT_SIZE, SECOND_OUTPUT_SIZE, FINAL_HIDDEN_SIZE, EMBEDDING_DIM, RNN_UNITS)
        
        for epoch_num in epoch_num_array:
          model.load_weights(f'./training_checkpoints/ckpt_{epoch_num}')
          model.build(tf.TensorShape([1, None]))

          total_num_except_0_1000 = len(test_rnn_data)
          correct = 0
          first_correct = 0
          second_correct = 0

          for i in tqdm(range(len(test_delta_output)), desc='check accuracy..'):
            inp = [test_rnn_data[i].tolist(), test_address_binary_input[i].tolist()]
            lstm_ans = generate_address(model, inp)
            if test_delta_output[i] in lstm_ans:
              correct += 1
              if test_delta_output[i] == lstm_ans[0]:
                first_correct += 1
              if test_delta_output[i] == lstm_ans[1]:
                second_correct += 1

          data['first_output_size'] = [FIRST_OUTPUT_SIZE]
          data['second_output_size'] = [SECOND_OUTPUT_SIZE]
          data['final_hidden_size'] = [FINAL_HIDDEN_SIZE]
          data['batch_size'] = [BATCH_SIZE]
          data['epoch_num'] = [epoch_num]
          data['train_accuracy'] = [history.history['accuracy'][epoch_num-1]]
          data['accuracy_except_0_and_1000'] = [round(correct/total_num_except_0_1000*100,2)]
          data['first_number_accuracy'] = [round(first_correct/total_num_except_0_1000*100,2)]
          data['second_number_accuracy'] = [round(second_correct/total_num_except_0_1000*100,2)]
          
          data_frame = pd.DataFrame(data)

          data_frame.to_csv(f"seq_length_{seq_length}.csv", mode='a', header=False)

          print("\naccuracy except 0 and 1000")
          print(round(correct/total_num_except_0_1000*100,2))
          print("first number accuracy")
          print(round(first_correct/total_num_except_0_1000*100,2))
          print("second number accuracy")
          print(round(second_correct/total_num_except_0_1000*100,2))

Cov = pd.read_csv(f"seq_length_{seq_length}.csv", name["first_output_size", "second_output_size", "final_hidden_size", "batch_size", "epoch_num", "train_accuracy", "accuracy_except_0_and_1000", "first_number_accuracy", "second_number_accuracy"])
Cov.to_csv(f"seq_length_{seq_length}.csv")
