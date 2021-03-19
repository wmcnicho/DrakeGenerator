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

class MyRNNCell(keras.layers.Layer):
    def __init__(self, output_size, hidden_units=10, **kwargs):
      super(MyRNNCell, self).__init__(**kwargs)
      self.hidden_units = hidden_units
      self.state_size = hidden_units
      self.output_size = output_size
      self.state_size = hidden_units

    def build(self, input_shape):
      self.batch_size = input_shape[0]
      self.w_xh = self.add_weight(shape=(input_shape[-1], self.hidden_units), initializer='random_normal', trainable=True, name="W_xh")
      self.w_hh = self.add_weight(shape=(self.hidden_units, self.hidden_units), initializer='random_normal', trainable=True, name="W_hh")
      self.w_hy = self.add_weight(shape=(self.hidden_units, self.output_size), initializer='random_normal', trainable=True, name="W_hy")
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
      return tf.zeros([batch_size, self.hidden_units])

    def call(self, inputs, states):
      if states is None:
        initial_states = tf.zeros([self.batch_size, self.hidden_units])
      else:
        initial_states = states[0]
      #h_t = w_hh.prev_h + w_xh.inputs
      # TODO check if 1d
      #tf.expand_dims(batch_input,0)
      input_to_h =  tf.matmul(inputs, self.w_xh)
      weighted_prev_state =  tf.matmul(initial_states, self.w_hh) 
      next_state = input_to_h + weighted_prev_state
      #h_t = activation(h_t)
      next_state = tf.keras.activations.tanh(next_state)
      #y_t = w_hy.h_t
      output = tf.matmul(next_state, self.w_hy)
      # return y_t, h_t
      return output, next_state

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
  # Step can be tuned as a hyper_parameter, prior training used seq_length
  step = seq_length
  for i in range(0, total_splits, step): 
    # We create seq_length datapoints for each character combo padding with 0
    for j in range(0, seq_length):
      start = i
      end = i+j+1
      xs = input_data_arr[start:end]
      xs = pad_sequences([xs], maxlen=seq_length, padding='pre')[0]
      y_seq = np.append(xs[1:], input_data_arr[end])
      # One hot encode output (this is not needed for 'Sparse categorical cross entropy' loss)
      #ys = tf.keras.utils.to_categorical(y_seq, num_classes=vocab_size)
      ys = y_seq
      agg_xs.append(xs.copy())
      agg_ys.append(ys.copy())
  print("Total Number of data points to use: {}".format(len(agg_xs)))
  numpy_xs = np.array(agg_xs)
  numpy_ys = np.array(agg_ys)
  return (numpy_xs, numpy_ys)

# Generate text
def generate_text(seed_text, model, id_to_char_fn, chars_to_gen=300, random=False):
  generated_text = seed_text
  chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=id_to_char_fn.get_vocabulary(), invert=True)
  prev_state = None
  for _ in range(chars_to_gen):    
      input_id = id_to_char_fn(tf.strings.unicode_split(generated_text, 'UTF-8'))
      padded_input_arr = pad_sequences([input_id], maxlen=seq_length, padding='pre')
      padded_input = tf.convert_to_tensor([padded_input_arr[0]]) # Wrap in a rank 1 tensor for batch compatiability
      # Unique to class impl
      prediction, state = model(padded_input, states=prev_state, return_state=True)
      prev_state = state
      predicted_logits = prediction[:, -1, :]

      if random is True:
        # No random sampling, only take the max
        predicted_ids = tf.argmax(predicted_logits[0])
        # Convert from token ids to characters
        str_from_ids = chars_from_ids(predicted_ids)
      else: 
        # Default behavior random categorical, which take a sample based off the weight of the logits
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)
        str_from_ids = tf.strings.reduce_join(chars_from_ids(predicted_ids), axis=-1)
      new_char = str_from_ids.numpy().decode('utf-8')[-1]# Jump through the conversion hoops and grab character
      generated_text += str_from_ids
  # Some conversion hoops for our text object
#   generated_text = generated_text.numpy().decode('utf-8')
  # A quick filter so we don't get a Parental Advisory
  generated_text = generated_text.numpy().decode('utf-8').replace('nigga', 'ninja').replace('Nigga', 'Ninja')
  return generated_text

## Main code paths
def train_model(output_path=None, debug=False):
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
    my_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    my_model.compile(loss=my_loss, optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'], run_eagerly=debug)
    # Fit throws a shape compatiability error
    my_model.fit(x=split_xs, y=split_ys, epochs=2, verbose=1, batch_size=64)
    # example_prediction = my_model(split_xs)
    # print(example_prediction.shape)
    print(my_model.summary())
    if output_path is not None:
      save_filepath = output_path + "/weights/class_model_weights.h5"
      print("Output path found saving to " + save_filepath)
      my_model.save_weights(save_filepath)
    return my_model

def main(model_path, do_train=False):
    """ Entry point """
    # TODO refactor to remove this load section, this is only here to get the conversion function which is expensive
    data = open('./archive/drake_lyrics.txt').read()
    print('Length of text: {} characters'.format(len(data)))
    vocab = sorted(set(data))
    ids_from_chars = preprocessing.StringLookup(vocabulary=list(vocab))
    vocab_size = len(ids_from_chars.get_vocabulary())

    if do_train:
        print("Training and saving model...")
        model = train_model(output_path=model_path)
    else:
        print("Loading model from disk...")
        model = DrakeLSTM(vocab_size, embedding_dim)
        # Dummy call needed to initalize variables strange TF quirk
        model(tf.convert_to_tensor([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,84,74,80]]))
        load_status = model.load_weights("./models/class_model/classes_bars/weights/class_model_weights.h5")
        print(load_status)
    
    # Generate text, this currently isn't compatiable with class approach
    print("Generating Bars...please wait")
    #seed_texts = ["[Verse]", "[Chorus]", "[Bridge]", "[Verse]", "[Chorus]", "[Bridge]"] 
    seed_texts = ["[Verse]", "you", "love", "boy", "I love", "I love you", "Kiki, ","Swanging"]
    for seed in seed_texts:
        num_chars=400
        output_text = generate_text(seed, model, ids_from_chars, chars_to_gen=num_chars)
        print(">>>>>>>>>>>>>>>>>>>>")
        print("Input seed: %s" % (seed))
        print("%d character generated sequence:\n%s\n" % (num_chars, output_text))
        print("<<<<<<<<<<<<<<<<<<<<")
        print("End of output for seed: %s" % (seed))

    #Hope you enjoyed :)
    return 0

if __name__ == "__main__":
    main(None, do_train=True)