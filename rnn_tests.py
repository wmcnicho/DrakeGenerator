import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from custom_models import MyRNNCell, MyCellModelWrapper, MyGRUCell, MyCrossEntropyLoss
import utils

def create_alphabet_data(seq_length=30):
    """
    Creates a dataset from the alphabet text file

    @return Tuple of (xs, ys, vocab_size) as a training set from the alaphabet sample file
    """
    data = open('./archive/alphabet2.txt').read()
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
    (xs, ys) = utils.split_data_new(all_ids.numpy(), vocab_size, seq_length)
    return (xs, ys, vocab_size, ids_from_chars)

def test_rnn_cell():
    tf.executing_eagerly()
    myCell = MyRNNCell(33)
    input_1 = tf.random.normal(shape=(33,))
    myCell.build(input_1.shape)
    y = myCell.call(input_1, None)
    y_2 = myCell(input_1, None)

def test_rnn_cell_batch():
    tf.executing_eagerly()
    myCell = MyRNNCell(33)
    input_1 = tf.random.normal(shape=(32, 33))
    myCell.build(input_1.shape)
    y = myCell.call(input_1, None)
    y_2 = myCell(input_1, None)

def test_gru_cell():
    tf.executing_eagerly()
    myCell = MyGRUCell(33)
    input_1 = tf.random.normal(shape=(33,))
    myCell.build(input_1.shape)
    y = myCell.call(input_1, None)
    y_2 = myCell(input_1, None)

def create_basic_rnn(output_size, cell):
    model = MyCellModelWrapper(cell)
    my_loss = tf.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(loss=my_loss, 
                    optimizer=keras.optimizers.Adam(lr=0.001),
                    metrics=['accuracy'],
                    run_eagerly=True)
    return model

def test_basic_rnn(doTrain=True, save_filename=None):
    seq_length = 30
    (xs, ys, vocab_size, ids_from_chars_fn) = create_alphabet_data(seq_length=30)
    cell = MyRNNCell(vocab_size)
    model = create_basic_rnn(vocab_size, cell)
    if doTrain:
        # test_input = keras.Input((vocab_size))
        #py_input = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,84,74,80]
        #test_input = tf.variable(shape=(None, 30))
        test_input = tf.random.normal(shape=(32, seq_length, vocab_size))
        y = model(test_input, training=False)
        # For this implementation we expect a one-hot encode as input
        my_loss = tf.losses.CategoricalCrossentropy(from_logits=True)
        my_optimizer = keras.optimizers.Adam(lr=0.001)
        test_pred = model.predict(xs[:32])
        test_loss = my_loss(test_pred, ys[:32])

        # test_input_2 = tf.random.normal(shape=(32, seq_length, 33))
        # test_pred = model(1)

        print(model.summary())
        model.fit(x=xs, y=ys, epochs=2, verbose=1)
        print(model.summary())
        if save_filename is not None:
            utils.save_model(save_filename, model, custom_dir='./models/test_model/simple_custom_rnn/')
    else: 
        utils.load_weights("alphabet_model_weights_3_epochs.h5", model, tf.TensorShape([32, 1, vocab_size]), custom_dir='./models/test_model/simple_custom_rnn/')
    num_chars=400
    seed_texts = ['abc','jkl','qrs','xyz', 'abb', 'jjkkll', 'xyy']
    for seed in seed_texts:
        output_text = utils.generate_text_one_h(seed, model, seq_length, ids_from_chars_fn, chars_to_gen=num_chars, random=True)
        print("Input seed: %s" % (seed))
        print("%d char sequence:\n%s\n" % (num_chars, output_text))

def test_custom_gru(doTrain=True, save_filename=None):
    seq_length = 30
    (xs, ys, vocab_size, ids_from_chars_fn) = create_alphabet_data(seq_length=30)
    cell = MyGRUCell(vocab_size)
    #cell = keras.layers.GRUCell(vocab_size)
    model = create_basic_rnn(vocab_size, cell)
    if doTrain:
        # test_input = keras.Input((vocab_size))
        #py_input = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,84,74,80]
        #test_input = tf.variable(shape=(None, 30))
        test_input = tf.random.normal(shape=(32, seq_length, 33))
        y = model(test_input)
        # For this implementation we expect a one-hot encode as input
        my_loss = tf.losses.CategoricalCrossentropy(from_logits=True)
        my_optimizer = keras.optimizers.Adam(lr=0.001)
        test_pred = model.predict(xs[:32])
        test_loss = my_loss(test_pred, ys[:32])

        # test_input_2 = tf.random.normal(shape=(32, seq_length, 33))
        # test_pred = model(1)

        print(model.summary())
        model.fit(x=xs, y=ys, epochs=2, verbose=1)
        print(model.summary())
        if save_filename is not None:
            utils.save_model(save_filename, model, custom_dir='./models/test_model/custom_gru/')
    else: 
        utils.load_weights("alphabet_model_weights_3_epochs.h5", model, tf.TensorShape([32, 1, vocab_size]), custom_dir='./models/test_model/custom_gru/')
    num_chars=100
    seed_texts = ['abc','jkl','qrs','xyz', 'abb', 'jjkkll', 'xyy']
    for seed in seed_texts:
        output_text = utils.generate_text_one_h(seed, model, seq_length, ids_from_chars_fn, chars_to_gen=num_chars, random=False)
        print("Input seed: %s" % (seed))
        print("%d char sequence:\n%s\n" % (num_chars, output_text))

def test_custom_loss():
    actual_output = tf.random.normal(shape=(32, 1, 33))
    test_output = tf.random.normal(shape=(32, 1, 33))
    keras_loss = tf.losses.CategoricalCrossentropy(from_logits=True)
    lib_loss_result = keras_loss(actual_output, test_output)
    my_loss = MyCrossEntropyLoss()
    my_loss_result = my_loss(actual_output, test_output)
    # Assert they are identical

def test_custom_training_loop():
    seq_length = 30
    (xs, ys, vocab_size, ids_from_chars_fn) = create_alphabet_data(seq_length=30)
    cell = MyGRUCell(vocab_size)
    #cell = keras.layers.GRUCell(vocab_size)
    model = create_basic_rnn(vocab_size, cell)
    my_loss = tf.losses.CategoricalCrossentropy(from_logits=True)
    my_optimizer = keras.optimizers.Adam(lr=0.001)
    for i in range(0, len(xs), 32):
        with tf.GradientTape() as tape:
            logits_batch = model(xs[i:i+32])
            loss_value = my_loss(ys[i:i+32], logits_batch)
            loss_value += sum(model.losses)
            #print(model.trainable_weights)
        grads = tape.gradient(loss_value, model.trainable_weights)
        my_optimizer.apply_gradients(zip(grads, model.trainable_variables))

def main():
    #test_rnn_cell()
    #test_rnn_cell_batch()
    #test_basic_rnn(doTrain=True, save_filename='alphabet_2_model_weights.h5')
    #test_gru_cell()
    #test_custom_gru(doTrain=False, save_filename='alphabet_model_weights_3_epochs.h5')
    #test_custom_loss()
    test_custom_training_loop()

if __name__ == "__main__":
    main()