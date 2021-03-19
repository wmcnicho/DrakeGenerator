import tensorflow as tf
from tensorflow import keras

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