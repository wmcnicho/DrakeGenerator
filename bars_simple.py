import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import os
import time
import matplotlib.pyplot as plt

# Hyperparameters
seq_length = 40
embedding_dim = 256
char_to_process = None # Set to None to use all

# Helper functions
def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

# Fancy graphs
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.show()

# Generate text
def generate_text(seed_text, model, id_to_char_fn, chars_to_gen=300):
  generated_text = seed_text
  chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=id_to_char_fn.get_vocabulary(), invert=True)
  for _ in range(chars_to_gen):    
      input_id = id_to_char_fn(tf.strings.unicode_split(generated_text, 'UTF-8'))
      padded_input_arr = pad_sequences([input_id], maxlen=seq_length, padding='pre')
      padded_input = padded_input_arr[0]
      prediction = model.predict_classes(padded_input)
      str_from_ids = tf.strings.reduce_join(chars_from_ids(prediction), axis=-1)    
      new_char = str_from_ids.numpy().decode('utf-8')[-1] # Jump through the conversion hoops and grab character
      generated_text += new_char
  return generated_text

# Manually splitting up the dataset,
# There is probably some built in methods to do this but why not recreate the wheel
# returns a tuple of vectors to labels
def split_data(input_data_arr, vocab_size, seq_length, total_splits=None):
  total_chars = len(input_data_arr)
  if total_splits is None:
    total_splits = total_chars-seq_length-1
  agg_xs = []
  agg_ys = []
  for i in range(0, total_splits, seq_length): # We could remove the step for more training data
    # We create seq_length datapoints for each character combo padding with 0
    for j in range(0, seq_length):
      start = i
      end = i+j+1
      xs = input_data_arr[start:end]
      xs = pad_sequences([xs], maxlen=seq_length, padding='pre')[0]
      ys = input_data_arr[end]
      ys_to_label = tf.keras.utils.to_categorical(ys, num_classes=vocab_size)
      agg_xs.append(xs.copy())
      agg_ys.append(ys_to_label)
  print("Total Number of data points to use: {}".format(len(agg_xs)))
  numpy_xs = np.array(agg_xs)
  numpy_ys = np.array(agg_ys)
  return (numpy_xs, numpy_ys)

def main(do_train=True):
  ## Open and pre-process the data
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

  # Output vocab mapping for sanity
  vocab_sample = list(range(0,vocab_size))
  tf_vocab = tf.convert_to_tensor(vocab_sample)
  mapped_vocab = chars_from_ids(tf_vocab).numpy()
  print(vocab_sample)
  print(mapped_vocab)

  if do_train:
    (split_xs, split_ys) = split_data(all_ids.numpy(), vocab_size, seq_length, char_to_process)

    ## Build the model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=seq_length))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)))
    model.add(tf.keras.layers.Dense(vocab_size, activation='softmax'))
    print(model.summary())

    adam = tf.keras.optimizers.Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    ## Train the model
    #earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    history = model.fit(x=split_xs, y=split_ys, epochs=10, verbose=1)
    # Uncomment to show dope graph
    #plot_graphs(history, 'accuracy')

    saved_model_dir = "./models/simple_model/"
    model_filename = 'simple_bars.h5'
    model.save(saved_model_dir+model_filename)
  else:
    print("Loading model from file.")
    model = tf.keras.models.load_model("./models/simple_model/simple_bars.h5")


  ## Generate text with model
  # Dank so we trained for 50 iterations on a slice of the data
  # Let's see what this model generates for a few seeds
  num_chars = 100
  seed_text = '[Verse]\n'
  output_text = generate_text(seed_text, model, ids_from_chars, chars_to_gen=num_chars)
  print("Input sequence was: %s" % (seed_text))
  print("%d character generated sequence:\n%s" % (num_chars, output_text))

  seed_text = 'boy'
  output_text = generate_text(seed_text, model, ids_from_chars, chars_to_gen=num_chars)
  print("Input sequence was: %s" % (seed_text))
  print("%d character generated sequence:\n%s" % (num_chars, output_text))

  seed_text = 'you'
  output_text = generate_text(seed_text, model, ids_from_chars, chars_to_gen=num_chars)
  print("Input sequence was: %s" % (seed_text))
  print("%d character generated sequence:\n%s" % (num_chars, output_text))

  seed_text = 'love'
  output_text = generate_text(seed_text, model, ids_from_chars, chars_to_gen=num_chars)
  print("Input sequence was: %s" % (seed_text))
  print("%d character generated sequence:\n%s" % (num_chars, output_text))


if __name__ == "__main__":
    main(do_train=False)