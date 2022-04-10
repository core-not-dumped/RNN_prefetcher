# -*- coding: utf-8 -*-
"""ex.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vSWv5h-BPEiYTe79J82aOu6Z0nX_Vuow
"""

import pandas as pd
import matplotlib.pyplot as plt

frame1 = pd.read_csv('prefetcher_2_seq_length_5_not_1.csv')
frame2 = pd.read_csv('prefetcher_2_seq_length_5_not_2.csv')
frame3 = pd.read_csv('prefetcher_2_seq_length_5_not_3.csv')

frame = pd.concat([frame1,frame2,frame3])

first_output_size_array = [64, 128, 256, 512]
for i in range(len(first_output_size_array)):
  print("first_output_size " + str(first_output_size_array[i]) + ": " + str(frame[frame['first_output_size'] == first_output_size_array[i]]['accuracy_except_0_and_1000'].mean()))

print()
  
second_output_size_array = [8, 16, 32]
for i in range(len(second_output_size_array)):
  print("second_output_size " + str(second_output_size_array[i]) + ": " + str(frame[frame['second_output_size'] == second_output_size_array[i]]['accuracy_except_0_and_1000'].mean()))

print()

final_hidden_size_array = [128, 256, 512]
for i in range(len(final_hidden_size_array)):
  print("final_hidden_size " + str(final_hidden_size_array[i]) + ": " + str(frame[frame['final_hidden_size'] == final_hidden_size_array[i]]['accuracy_except_0_and_1000'].mean()))

print()

batch_size_array = [32, 64, 128]
for i in range(len(batch_size_array)):
  print("batch_size " + str(batch_size_array[i]) + ": " + str(frame[frame['batch_size'] == batch_size_array[i]]['accuracy_except_0_and_1000'].mean()))

print()

epoch_num_array = [10, 15, 20, 25, 30]
for i in range(len(epoch_num_array)):
  print("batch_size " + str(epoch_num_array[i]) + ": " + str(frame[frame['epoch_num'] == epoch_num_array[i]]['accuracy_except_0_and_1000'].mean()))

print(frame1.iloc[:,7:-2])

frame1 = frame1.iloc[:,7:-2]
frame2 = frame2.iloc[:,7:-2]
frame3 = frame3.iloc[:,7:-2]

frame1.plot()
frame2.plot()
frame3.plot()

frame1.plot.hist()
frame2.plot.hist()
frame3.plot.hist()



"""## access pattern"""

from google.colab import drive
drive.mount('/content/drive')

!pwd

# Commented out IPython magic to ensure Python compatibility.
# %cd drive/MyDrive/Colab\ Notebooks

#f = open(g.text_file + ".txt", 'r')
f1 = open('/content/drive/MyDrive/Colab Notebooks/dataset/' + '459.GemsFDTD-s1' + '.txt', 'r')
f2 = open('/content/drive/MyDrive/Colab Notebooks/dataset/' + '473.astar-s0' + '.txt', 'r')

load_address1 = list()
load_address2 = list()

f1.seek(0, 0)
f2.seek(0, 0)
while True:
  line1 = f1.readline()
  if not line1: break
  split_line1 = line1.split(', ')
  load_address1.append(split_line1[2])

while True:
  line2 = f2.readline()
  if not line2: break
  split_line2 = line2.split(', ')
  load_address2.append(split_line2[2])
  
for i in range(len(load_address1)):
  load_address1[i] = int(load_address1[i], 16)//64

for i in range(len(load_address2)):
  load_address2[i] = int(load_address2[i], 16)//64

load_address = load_address1

import matplotlib.pyplot as plt

delta = list()
for i in range(len(load_address)-1):
  delta.append(load_address[i+1] - load_address[i])

for i in range(len(delta)):
  if delta[i] > 2000:
    delta[i] = 0
  if delta[i] < -2000:
    delta[i] = 0

plt.hist(delta, bins = 50)

import numpy as np

load_address1 = np.array(load_address1)
load_address2 = np.array(load_address2)

load_address1 = load_address1 >> 8
load_address2 = load_address2 >> 8

p = 1

x1 = load_address1[:len(load_address1)//p]
y1 = np.arange(len(load_address1)//p)
plt.scatter(y1, x1, s = 0.001)

x2 = load_address2[:len(load_address2)//p]
y2 = np.arange(len(load_address2)//p)
#plt.scatter(y2, x2, s = 0.001)

