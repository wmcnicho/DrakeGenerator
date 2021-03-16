import tensorflow as tf
from bars_classes import MyRNNCell
from tensorflow import keras
from bars_classes import split_data
from tensorflow.keras.layers.experimental import preprocessing

def create_basic_rnn(output_size):
    cell = MyRNNCell(26)
    my_rnn = keras.layers.RNN(cell)

    model = keras.models.Sequential(my_rnn)
    my_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(loss=my_loss, 
                    optimizer=keras.optimizers.Adam(lr=0.001),
                    metrics=['accuracy'],
                    run_eagerly=True)
    return model

def create_alphabet_data():
    """
    Creates a dataset from the alphabet text file

    @return Tuple of (xs, ys, vocab_size) as a training set from the alaphabet sample file
    """
    data = open('./archive/alphabet.txt').read()
    #print('Length of text: {} characters'.format(len(data)))
    vocab = sorted(set(data))
    # This function as variable setup is weird to me but whatever
    ids_from_chars = preprocessing.StringLookup(vocabulary=list(vocab))
    chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True)

    # Preprocess the text into characters
    all_ids = ids_from_chars(tf.strings.unicode_split(data, 'UTF-8'))
    vocab_size = len(ids_from_chars.get_vocabulary())

    # Sanity tests
    vocab_sample = list(range(0,vocab_size))
    tf_vocab = tf.convert_to_tensor(vocab_sample)
    mapped_vocab = chars_from_ids(tf_vocab).numpy()

    seq_length = 30
    # Warning: This is an untested function used as a test dependency
    (xs, ys) = split_data(all_ids.numpy(), vocab_size, seq_length)
    return (xs, ys, vocab_size)


def test_basic_rnn():
    (xs, ys, vocab_size) = create_alphabet_data()
    model = create_basic_rnn(vocab_size)
    model.fit(x=xs, y=ys, epochs=10, verbose=1)
    print(model.summary())


def main():
    test_basic_rnn()

if __name__ == "__main__":
    main()