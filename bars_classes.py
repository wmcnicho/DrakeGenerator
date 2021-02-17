import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
seq_length = 40
embedding_dim = 256
char_to_process = None # Set to None to use all

class DrakeLSTM(keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units=150):
        super().__init__()
        self.embed_layer = keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
        #self.lstm = keras.layers.LSTM(lstm_units, return_state=True)
        self.reshape_layer = keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embed_layer(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.reshape_layer(x, training=training)

        if return_state:
            return x, states
        else:
            return x

# TODO This is copy-pasta from bars_simple.py could be extracted to a separate utilities file
# Helper functions
def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

def text_from_ids(ids, char_to_id_fn):
  return tf.strings.reduce_join(char_to_id_fn(ids), axis=-1)

# Fancy graphs
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.show()

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
      y_seq = np.append(xs[1:], input_data_arr[end])
      ys = tf.keras.utils.to_categorical(y_seq, num_classes=vocab_size)
      agg_xs.append(xs.copy())
      agg_ys.append(ys.copy())
  print("Total Number of data points to use: {}".format(len(agg_xs)))
  numpy_xs = np.array(agg_xs)
  numpy_ys = np.array(agg_ys)
  return (numpy_xs, numpy_ys)

# Generate text
def generate_text(seed_text, model, id_to_char_fn, chars_to_gen=300):
  generated_text = seed_text
  for _ in range(chars_to_gen):    
      input_id = id_to_char_fn(tf.strings.unicode_split(generated_text, 'UTF-8'))
      padded_input_arr = pad_sequences([input_id], maxlen=seq_length, padding='pre')
      padded_input = padded_input_arr[0]
      # Unique to class impl
      prediction = model.predict(padded_input)
      predicted_logits = prediction[:, -1, :]
      predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
      predicted_ids = tf.squeeze(predicted_ids, axis=-1)
      # Convert from token ids to characters
      chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=id_to_char_fn.get_vocabulary(), invert=True)
      str_from_ids = tf.strings.reduce_join(chars_from_ids(predicted_ids), axis=-1)
      new_char = str_from_ids.numpy().decode('utf-8')[-1]# Jump through the conversion hoops and grab character
      generated_text += new_char
  return generated_text

## Main code paths
def train_model(save=False, output_path="./models"):
    """ Codepath to process input and train (as opposed to load up and generate)"""
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

    print("Splitting file into dataset")
    (split_xs, split_ys) = split_data(all_ids.numpy(), vocab_size, seq_length, total_splits=char_to_process)
    
    # Create the Model
    my_model = DrakeLSTM(vocab_size, embedding_dim)
    my_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'], run_eagerly=True)
    # Fit throws a shape compatiability error
    my_model.fit(x=split_xs, y=split_ys, epochs=5, verbose=1)
    # example_prediction = my_model(split_xs)
    # print(example_prediction.shape)
    print(my_model.summary())
    return my_model

def main(do_train=True):
    """ Entry point """
    # TODO refactor to remove this load section, this is only here to get the conversion function which is expensive
    data = open('./archive/drake_lyrics.txt').read()
    print('Length of text: {} characters'.format(len(data)))
    vocab = sorted(set(data))
    ids_from_chars = preprocessing.StringLookup(vocabulary=list(vocab))

    # TODO implement save and load with saved_model, looks easy
    # if train:
    #     model = train()
    # else:
    #     # Load model
    model = train_model()

    
    # Generate text, this currently isn't compatiable with class approach
    print("Generating Bars...please wait")
    seed_texts = ["[Verse]", "you", "love", "boy", "I love", "I love you", "Kiki, ","Swanging"]
    for seed in seed_texts:
        num_chars=100
        output_text = generate_text(seed, model, ids_from_chars, chars_to_gen=num_chars)
        print("Input sequence: %s" % (seed))
        print("%d character generated sequence:\n%s\n" % (num_chars, output_text))

    #Hope you enjoyed :)
    return 0

if __name__ == "__main__":
    main(do_train=True)