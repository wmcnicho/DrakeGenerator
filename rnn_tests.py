import tensorflow as tf
from bars_classes import MyRNNCell
from tensorflow import keras
from bars_classes import split_data
from tensorflow.keras.layers.experimental import preprocessing

class MyCellModelWrapper(keras.Model):
    def __init__(self, cell):
        super().__init__()
        self.rnn = keras.layers.RNN(cell, return_state=True)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        # if states is None:
        #     states = self.rnn.get_initial_state(x)
        x, states = self.rnn(x)
        if return_state:
            return x, states
        else:
            return x

def create_basic_rnn(output_size):
    cell = MyRNNCell(output_size)
    model = MyCellModelWrapper(cell)
    my_loss = tf.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(loss=my_loss, 
                    optimizer=keras.optimizers.Adam(lr=0.001),
                    metrics=['accuracy'])
    model.run_eagerly = True
    return model

def split_data_new(input_data_arr, vocab_size, seq_length, total_splits=None):
  total_chars = len(input_data_arr)
  if total_splits is None:
    total_splits = total_chars-seq_length-1
  agg_xs = []
  agg_ys = []
  for i in range(seq_length, total_splits-1):
    start = i - seq_length
    end = i
    xs = input_data_arr[start:end]
    ys = input_data_arr[end+1]
    #Copies are expensive let's try to avoid this
    #agg_xs.append(xs.copy())
    #agg_ys.append(ys.copy())
    agg_xs.append(xs)
    agg_ys.append(ys)
  print("Total Number of data points to use: {}".format(len(agg_xs)))
  oh_xs = tf.keras.utils.to_categorical(agg_xs, num_classes=vocab_size)
  oh_ys = tf.keras.utils.to_categorical(agg_ys, num_classes=vocab_size)
  return (oh_xs, oh_ys)

def create_alphabet_data(seq_length=30):
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

    # Warning: This is an untested function used as a test dependency
    (xs, ys) = split_data_new(all_ids.numpy(), vocab_size, seq_length)
    return (xs, ys, vocab_size)

def test_rnn_cell():
    tf.executing_eagerly()
    myCell = MyRNNCell(33)
    input_1 = tf.random.normal(shape=(32, 33))
    myCell.build(input_1.shape)
    y = myCell.call(input_1, None)
    y_2 = myCell(input_1, None)

def test_basic_rnn():
    seq_length = 30
    (xs, ys, vocab_size) = create_alphabet_data(seq_length=30)
    model = create_basic_rnn(vocab_size)
    # test_input = keras.Input((vocab_size))
    #py_input = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,84,74,80]
    #test_input = tf.variable(shape=(None, 30))
    test_input = tf.random.normal(shape=(32, seq_length, 33))
    y = model(test_input)
    # For this implementation we expect a one-hot encode as input
    my_loss = tf.losses.CategoricalCrossentropy(from_logits=True)
    test_pred = model.predict(xs[:32])
    test_loss = my_loss(test_pred, ys[:32])
    model.fit(x=xs, y=ys, epochs=10, verbose=1)
    print(model.summary())


def main():
    #test_rnn_cell()
    test_basic_rnn()

if __name__ == "__main__":
    main()