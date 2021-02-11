import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import os
import time
import matplotlib.pyplot as plt

# Hyperparameters
seq_length = 100
embedding_dim = 256

# Helper functions
def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

def text_from_ids(ids):
  return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

# Fancy graphs
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.show()

# Manually splitting up the dataset,
# There is probably some built in methods to do this but why not recreate the wheel
# returns a tuple of vectors to labels
def split_data(input_data_arr, split_size, total_splits=None):
  total_chars = len(input_data_arr)
  if total_splits is None:
    total_splits = total_chars-split_size-1
  agg_xs = []
  agg_ys = []
  for i in range(0, total_splits):
    xs = input_data_arr[i:i+split_size]
    ys = input_data_arr[i+1 :i+split_size+1]
    # Do we need to one-hot encode the output?
    ys_to_label = tf.keras.utils.to_categorical(ys[-1], num_classes=vocab_size)
    agg_xs.append(xs.copy())
    agg_ys.append(ys_to_label)
  numpy_xs = np.array(agg_xs)
  # Do we need to one-hot encode the output?
  #ys_to_label = tf.keras.utils.to_categorical(agg_ys, num_classes=vocab_size)
  numpy_ys = np.array(agg_ys)
  #ys_to_label = tf.keras.utils.to_categorical(agg_ys, num_classes=vocab_size)
  #numpy_ys = np.array(ys_to_label)
  return (numpy_xs, numpy_ys)

# Verified this works with alphabet now lets make things more interesting
# data = open('./archive/alphabet.txt').read()
data = open('./archive/drake_lyrics.txt').read()
print('Length of text: {} characters'.format(len(data)))

vocab = sorted(set(data))

# This function as variable setup is weird to me but whatever
ids_from_chars = preprocessing.StringLookup(vocabulary=list(vocab))
chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True)

# Preprocess the text into characters
all_ids = ids_from_chars(tf.strings.unicode_split(data, 'UTF-8'))
vocab_size = len(ids_from_chars.get_vocabulary())

(split_xs, split_ys) = split_data(all_ids.numpy(), seq_length)

# This might overcomplicate things but we'll leave it for now to save time
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)
dataset = sequences.map(split_input_target)

# # Batch the dataset
# BATCH_SIZE = 50
# dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

# This assignment is sketch, complicated by the dataset pipeline tf paradigm
for input_ex, output_ex in dataset.take(1):
    print(input_ex)
    print(output_ex)

xs = input_ex
labels = output_ex
# One-hot encode the output
ys = tf.keras.utils.to_categorical(labels, num_classes=vocab_size)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=seq_length))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)))
model.add(tf.keras.layers.Dense(vocab_size, activation='softmax'))
print(model.summary())

adam = tf.keras.optimizers.Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
#history = model.fit(xs, ys, epochs=100, verbose=1)
history = model.fit(x=split_xs, y=split_ys, epochs=10, verbose=1)
plot_graphs(history, 'accuracy')

# Dank so we trained for 50 iterations on a slice of the data
# Let's see what this model generates for a few seeds
seed_text = 'you'
chars_to_gen = 300
generated_text = seed_text

for _ in range(chars_to_gen):    
    input_id = ids_from_chars(tf.strings.unicode_split(generated_text, 'UTF-8'))
    padded_input_arr = pad_sequences([input_id], maxlen=seq_length, padding='pre')
    padded_input = padded_input_arr[0]
    prediction = model.predict_classes(padded_input)
    new_char = text_from_ids(prediction).numpy().decode('utf-8')[-1] # Jump through the conversion hoops and grab character
    generated_text += new_char
    # predicted = model.predict_classes()

print("Input sequence was: %s" % (seed_text))
print("%d character generated sequence:\n%s" % (chars_to_gen, generated_text))