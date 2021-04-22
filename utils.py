import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import numpy as np

output_dir = "./models/class_model/classes_bars/"
weights_dir = "weights/"

"""
    The always necessary but quickly bloated utility class
"""

def save_model(file_name, model, custom_dir=None):
    directory = output_dir if custom_dir is None else custom_dir
    save_filepath = directory + weights_dir + file_name
    print("Output path found saving to " + save_filepath)
    model.save_weights(save_filepath)
    return save_filepath

def text_from_ids(ids, char_to_id_fn):
  return tf.strings.reduce_join(char_to_id_fn(ids), axis=-1)

def load_weights(file_name, model, input_shape, custom_dir=None):
  directory = output_dir if custom_dir is None else custom_dir
  # Dummy call needed to initalize variables strange TF quirk, maybe to call build?
  # tf.convert_to_tensor([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,84,74,80]])
  sample_input = tf.zeros(input_shape)
  model(sample_input)
  load_status = model.load_weights(directory + weights_dir + file_name)
  print(load_status)
  return None

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

def split_data_new(input_data_arr, vocab_size, seq_length, total_splits=None):
  # This data split works slightly differently but is much faster
  # Output is one-hot encoded to the output
  total_chars = len(input_data_arr)
  if total_splits is None:
    total_splits = total_chars-seq_length-1
  agg_xs = []
  agg_ys = []
  for i in range(seq_length, total_splits-1):
    start = i - seq_length
    end = i
    xs = input_data_arr[start:end]
    ys = input_data_arr[start+1:end+1]
    agg_xs.append(xs)
    agg_ys.append(ys)
  print("Total Number of data points to use: {}".format(len(agg_xs)))
  oh_xs = tf.keras.utils.to_categorical(agg_xs, num_classes=vocab_size)
  oh_ys = tf.keras.utils.to_categorical(agg_ys, num_classes=vocab_size)
  return (oh_xs, oh_ys)

# Text generation methods to extract sequences from models
def generate_text_one_h(seed_text, model, seq_length, char_to_id_fn, chars_to_gen=300, random=True):
  output_text = seed_text
  chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=char_to_id_fn.get_vocabulary(), invert=True)
  vocab_size = len(char_to_id_fn.get_vocabulary())
  prev_state = None
  input_string = seed_text
  for _ in range(chars_to_gen):    
    input_id = char_to_id_fn(tf.strings.unicode_split(input_string, 'UTF-8'))
    input_logits = tf.keras.utils.to_categorical(input_id, num_classes=vocab_size)
    input_logits = tf.expand_dims(input_logits, 0) # Add a dummy batch
    predicted_logits, prev_state =  model(input_logits, states=prev_state, return_state=True)
    predicted_logits = predicted_logits[:, -1, :]
    if random is True:
      # Default behavior random categorical, which take a sample based off the weight of the logits
      predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
      predicted_ids = tf.squeeze(predicted_ids, axis=-1)
      # TODO this could probably be cleaner
      if predicted_ids[0] == 0 or predicted_ids[0] == 1:
          # this is the empty/unknown characters, we should ignore
          #print('bad char detected')
          continue
      predicted_char = tf.strings.reduce_join(chars_from_ids(predicted_ids), axis=-1)
    else:
      # No random sampling, only take the max
      predicted_ids = tf.argmax(predicted_logits[0])
      # Convert from token ids to characters
      predicted_char = chars_from_ids(predicted_ids)
    input_string = predicted_char
    output_text += predicted_char
  return output_text.numpy().decode('utf-8')

# Generate text
def generate_text(seed_text, model, seq_length, char_to_id_fn, chars_to_gen=300, random=False):
  generated_text = seed_text
  chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=char_to_id_fn.get_vocabulary(), invert=True)
  prev_state = None
  for _ in range(chars_to_gen):    
      input_id = char_to_id_fn(tf.strings.unicode_split(generated_text, 'UTF-8'))
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
      decode = str_from_ids.numpy().decode('utf-8')
      new_char = decode[-1] if len(decode) > 0 else decode# Jump through the conversion hoops and grab character
      generated_text += new_char
  # Some conversion hoops for our text object
  #   generated_text = generated_text.numpy().decode('utf-8')
  # A quick filter so we don't get a Parental Advisory
  generated_text = generated_text.numpy().decode('utf-8').replace('nigga', 'ninja').replace('Nigga', 'Ninja')
  return generated_text