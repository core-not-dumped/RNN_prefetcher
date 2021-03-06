# -*- coding: utf-8 -*-
"""auto_rnn_prefetcher_1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wr6O0Wq29RsNgbwmGStVnfhVSkeBEl2H
"""

from google.colab import drive
drive.mount('/content/drive')

!pwd

# Commented out IPython magic to ensure Python compatibility.
# %cd drive/MyDrive/Colab\ Notebooks

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import globalobject as g
from keras.datasets import imdb
from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import shutil

seq_length = g.seq_length
epoch = g.epoch
MAXLEN = g.MAXLEN
DENSE_HIDDEN_SIZE = g.DENSE_HIDDEN_SIZE
train_model_percentage = g.train_model_percentage

ADDRESS_PREDICT_SIZE = g.ADDRESS_PREDICT_SIZE
EMBEDDING_DIM = g.EMBEDDING_DIM
RNN_UNITS = g.RNN_UNITS

BUFFER_SIZE = g.BUFFER_SIZE

text_file = g.text_file

train_rnn_data = np.load('/content/drive/MyDrive/Colab Notebooks/dataset/' + g.text_file + '_train_rnn_data.npy')
test_rnn_data = np.load('/content/drive/MyDrive/Colab Notebooks/dataset/' + g.text_file + '_test_rnn_data.npy')
train_delta_output = np.load('/content/drive/MyDrive/Colab Notebooks/dataset/' + g.text_file + '_train_delta_output.npy')
test_delta_output = np.load('/content/drive/MyDrive/Colab Notebooks/dataset/' + g.text_file + '_test_delta_output.npy')

from keras.layers import Input, Dense, Embedding, LSTM
from keras.models import Model

def build_model(address_predict_size, embedding_dim, rnn_units):
  input_x = Input(shape=(seq_length,))
  x = Embedding(address_predict_size, embedding_dim)(input_x)
  x = LSTM(rnn_units, recurrent_initializer='glorot_uniform') (x)
  x = Dense(address_predict_size, activation="sigmoid") (x)
  model = Model(inputs=input_x, outputs=x)
  return model

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits)

def generate_address(model, start_address):
  # Evaluation step (generating address using the learned model)

  # Number of characters to generate
  num_generate = 2
  input_eval = tf.expand_dims(start_address, 0)

  # Empty string to store our results
  generated = []

  # Here batch size == 1
  model.reset_states()
  predictions = model(input_eval)

  # using argmax
  predictions_np = predictions.numpy()
  predictions_np = np.squeeze(predictions_np,axis=0)
  predictions_np = np.argsort(predictions_np)
  generated = predictions_np [-num_generate:]

  return generated

BATCH_SIZE_array = [32, 64, 128]

data = dict()
continue_check = 0

if os.path.isfile(f'/content/drive/MyDrive/Colab Notebooks/prefetcher_1_seq_length_{seq_length}.csv') == True:
  continue_check = 1
  before_data = pd.read_csv(f"/content/drive/MyDrive/Colab Notebooks/prefetcher_1_seq_length_{seq_length}.csv", names=["batch_size", "epoch_num", "train_accuracy", "accuracy_except_0_and_1000", "first_number_accuracy", "second_number_accuracy"])

for BATCH_SIZE in BATCH_SIZE_array:

  if os.path.isdir("./training_checkpoints") == True:
    shutil.rmtree("./training_checkpoints")

  print("\n\nnew model start!!")
  print(f"{BATCH_SIZE}\n")

  epoch_num_array = [10, 15, 20, 25, 30]

  if continue_check == 1:
    tmp = before_data.loc[before_data["batch_size"] == BATCH_SIZE]
    if tmp.shape[0] == len(epoch_num_array):
      continue
    else:
      continue_check = 0
      epoch_num_array = epoch_num_array[tmp.shape[0]:]

  model = build_model(ADDRESS_PREDICT_SIZE,EMBEDDING_DIM, RNN_UNITS)
      
  model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
        
  checkpoint_dir = './training_checkpoints'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

  checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
        
  history = model.fit(train_rnn_data, train_delta_output, batch_size=BATCH_SIZE, epochs=epoch, callbacks=[checkpoint_callback])

  model = build_model(ADDRESS_PREDICT_SIZE,EMBEDDING_DIM, RNN_UNITS)
        
  for epoch_num in epoch_num_array:
    model.load_weights(f'./training_checkpoints/ckpt_{epoch_num}')
    model.build(tf.TensorShape([1, None]))

    total_num_except_0_1000 = len(test_rnn_data)
    correct = 0
    first_correct = 0
    second_correct = 0

    for i in tqdm(range(len(test_delta_output)), desc='check accuracy..'):
      inp = test_rnn_data[i]
      lstm_ans = generate_address(model, inp)
      if test_delta_output[i] in lstm_ans:
        correct += 1
        if test_delta_output[i] == lstm_ans[0]:
          first_correct += 1
        if test_delta_output[i] == lstm_ans[1]:
          second_correct += 1

    data['batch_size'] = [BATCH_SIZE]
    data['epoch_num'] = [epoch_num]
    data['train_accuracy'] = [round(history.history['accuracy'][epoch_num-1]*100,2)]
    data['accuracy_except_0_and_1000'] = [round(correct/total_num_except_0_1000*100,2)]
    data['first_number_accuracy'] = [round(first_correct/total_num_except_0_1000*100,2)]
    data['second_number_accuracy'] = [round(second_correct/total_num_except_0_1000*100,2)]
            
    data_frame = pd.DataFrame(data)

    data_frame.to_csv(f"/content/drive/MyDrive/Colab Notebooks/prefetcher_1_seq_length_{seq_length}.csv", mode='a', header=False)

    print("\naccuracy except 0 and 1000")
    print(round(correct/total_num_except_0_1000*100,2))
    print("first number accuracy")
    print(round(first_correct/total_num_except_0_1000*100,2))
    print("second number accuracy")
    print(round(second_correct/total_num_except_0_1000*100,2))

Cov = pd.read_csv(f"/content/drive/MyDrive/Colab Notebooks/prefetcher_1_seq_length_{seq_length}.csv", names = ["batch_size", "epoch_num", "train_accuracy", "accuracy_except_0_and_1000", "first_number_accuracy", "second_number_accuracy"])
Cov.to_csv(f"/content/drive/MyDrive/Colab Notebooks/prefetcher_1_seq_length_{seq_length}.csv")

