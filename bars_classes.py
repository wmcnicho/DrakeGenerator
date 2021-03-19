import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from custom_models import DrakeLSTM

import numpy as np
import utils

# Hyperparameters
seq_length = 40
embedding_dim = 256
char_to_process = None # Set to None to use all

## Main code paths
def train_model(file_name=None, debug=False):
    """ Codepath to process input and train (as opposed to load up and generate)"""
    # Load Data
    data = open('./archive/drake_lyrics.txt').read()
    print('Length of text: {} characters'.format(len(data)))
    vocab = sorted(set(data))
    # Preprocess the text into integers
    ids_from_chars = preprocessing.StringLookup(vocabulary=list(vocab))
    chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True)
    all_ids = ids_from_chars(tf.strings.unicode_split(data, 'UTF-8'))
    vocab_size = len(ids_from_chars.get_vocabulary())

    # Sanity Check: output vocab mapping
    vocab_sample = list(range(0,vocab_size))
    tf_vocab = tf.convert_to_tensor(vocab_sample)
    mapped_vocab = chars_from_ids(tf_vocab).numpy()
    print(vocab_sample)
    print(mapped_vocab)

    # Creating dataset from pre-processed text
    print("Splitting file into dataset")
    (split_xs, split_ys) = utils.split_data(all_ids.numpy(), vocab_size, seq_length, total_splits=char_to_process)
    
    # Create the Model
    my_model = DrakeLSTM(vocab_size, embedding_dim)
    my_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    my_model.compile(loss=my_loss, optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'], run_eagerly=debug)

    # Train the model
    # TODO run this in a gradient tape loop and play with batch randomization
    my_model.fit(x=split_xs, y=split_ys, epochs=2, verbose=1, batch_size=64)
    
    print(my_model.summary())
    if file_name is not None:
      utils.save_model(file_name, my_model)
    return (my_model, vocab)

def main(save_filename=None,  load_filename="class_model_weights.h5", do_train=False):
    """ Entry point """
    if do_train:
      print("Training and saving model...")
      (model, vocab) = train_model(file_name=save_filename)
      ids_from_chars = preprocessing.StringLookup(vocabulary=list(vocab))
      vocab_size = len(ids_from_chars.get_vocabulary())
    else:
        if load_filename is None:
          print("ERROR load file name not provided and training flag set to false, no model can be used")
          return 1
        # TODO Somehow this vocab should be accessible without needed to read and process this data
        data = open('./archive/drake_lyrics.txt').read()
        print('Length of text: {} characters'.format(len(data)))
        vocab = sorted(set(data))
        ids_from_chars = preprocessing.StringLookup(vocabulary=list(vocab))
        vocab_size = len(ids_from_chars.get_vocabulary())
        print("Loading model from disk...")
        model = DrakeLSTM(vocab_size, embedding_dim)
        utils.load_weights(load_filename, model)
    
    # Generate text, this currently isn't compatiable with class approach
    print("Generating Bars...please wait")
    #seed_texts = ["[Verse]", "[Chorus]", "[Bridge]", "[Verse]", "[Chorus]", "[Bridge]"] 
    seed_texts = ["[Verse]", "you", "love", "boy", "I love", "I love you", "Kiki, ","Swanging"]
    for seed in seed_texts:
        num_chars=400
        output_text = utils.generate_text(seed, model, seq_length, ids_from_chars, chars_to_gen=num_chars)
        print(">>>>>>>>>>>>>>>>>>>>")
        print("Input seed: %s" % (seed))
        print("%d character generated sequence:\n%s\n" % (num_chars, output_text))
        print("<<<<<<<<<<<<<<<<<<<<")
        print("End of output for seed: %s" % (seed))
    #Hope you enjoyed :)
    return 0

if __name__ == "__main__":
    main(load_filename="class_model_weights.h5", do_train=True)