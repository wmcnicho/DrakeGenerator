import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import os
import time

# Magic numbers, I'm not sure why
seq_length = 175
embedding_dim = 256

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

data = open('./archive/drake_lyrics.txt').read()
print('Length of text: {} characters'.format(len(data)))

vocab = sorted(set(data))
vocab_size = len(vocab)+2 # this +2 is a moonshot

# This setup is a bit strange to store as variables to me
ids_from_chars = preprocessing.StringLookup(
    vocabulary=list(vocab))
chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True)

all_ids = ids_from_chars(tf.strings.unicode_split(data, 'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
for ids in ids_dataset.take(10):
    print(chars_from_ids(ids).numpy().decode('utf-8'))

sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)
dataset = sequences.map(split_input_target)



for input_example, target_example in  dataset.take(1):
    print("Unmapped data:")
    print("Input :", input_example.numpy())
    print("Target:", target_example.numpy())
    print("Mapped data:")
    print("Input :", chars_from_ids(input_example).numpy())
    print("Target:", chars_from_ids(target_example).numpy())

print(dataset)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)))
model.add(tf.keras.layers.Dense(vocab_size))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
print(model.summary())

model.fit(dataset, epochs=100, verbose=1)
print(model)
