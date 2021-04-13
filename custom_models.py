import tensorflow as tf
from tensorflow import keras
from numpy import log2

class DrakeGRUSequential(keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units=150):
        super().__init__()
        self.embed_layer = keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
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

class MyCellModelWrapper(keras.Model):
    def __init__(self, cell):
        super().__init__()
        self.rnn = keras.layers.RNN(cell, return_state=True, return_sequences=True)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        if states is None:
            states = self.rnn.get_initial_state(x)
        x, states = self.rnn(x, initial_state=states, training=training) # x= (batch_size, seq_length, vocab_size), states= (batch_sizes, hidden)
        if return_state:
            return x, states
        else:
            return x

class MyRNNCell(keras.layers.Layer):
    def __init__(self, output_size, hidden_units=150, **kwargs):
      super(MyRNNCell, self).__init__(**kwargs)
      self.hidden_units = hidden_units
      self.output_size = output_size
      self.state_size = hidden_units

    def build(self, input_shape):
      self.w_xh = self.add_weight(shape=(input_shape[-1], self.hidden_units), initializer='random_normal', trainable=True, name="W_xh")
      self.w_hh = self.add_weight(shape=(self.hidden_units, self.hidden_units), initializer='random_normal', trainable=True, name="W_hh")
      self.w_hy = self.add_weight(shape=(self.hidden_units, self.output_size), initializer='random_normal', trainable=True, name="W_hy")
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
      batch = batch_size if batch_size is not None else inputs.shape[0]
      return tf.zeros((batch, self.hidden_units))

    def call(self, inputs, states):
      if len(inputs.shape) is 1:
        inputs = tf.expand_dims(inputs, 0)
      if states is None:
        initial_states = self.get_initial_state(inputs)
      else:
        initial_states = states[0]
      #h_t = w_hh.prev_h + w_xh.inputs
      input_to_h =  tf.matmul(inputs, self.w_xh)
      weighted_prev_state =  tf.matmul(initial_states, self.w_hh) 
      next_state = input_to_h + weighted_prev_state
      #h_t = activation(h_t)
      next_state = tf.keras.activations.tanh(next_state)
      #y_t = w_hy.h_t
      output = tf.matmul(next_state, self.w_hy)
      # return y_t, h_t
      return output, next_state

class MyGRUCell(keras.layers.Layer):
    def __init__(self, output_size, hidden_units=10, **kwargs):
      super(MyGRUCell, self).__init__(**kwargs)
      self.hidden_units = hidden_units
      self.output_size = output_size
      self.state_size = hidden_units

    def build(self, input_shape):
      self.w_z = self.add_weight(shape=(input_shape[-1], self.hidden_units), initializer='random_normal', trainable=True, name="Input Update Weights")
      self.h_z = self.add_weight(shape=(self.hidden_units, self.hidden_units), initializer='random_normal', trainable=True, name="Hidden Update Weights")
      self.w_r = self.add_weight(shape=(input_shape[-1], self.hidden_units), initializer='random_normal', trainable=True, name="Input Reset Weights")
      self.h_r = self.add_weight(shape=(self.hidden_units, self.hidden_units), initializer='random_normal', trainable=True, name="Hidden Reset Weights")
      self.w_mc = self.add_weight(shape=(input_shape[-1], self.hidden_units), initializer='random_normal', trainable=True, name="Input Memory Context Weights")
      self.h_mc = self.add_weight(shape=(self.hidden_units, self.hidden_units), initializer='random_normal', trainable=True, name="Hidden Memory Context Weights")
      self.w_y = self.add_weight(shape=(self.hidden_units, self.output_size), initializer='random_normal', trainable=True, name="Hidden to outputs")
      
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
      batch = batch_size if batch_size is not None else inputs.shape[0]
      return tf.zeros((batch, self.hidden_units))
    
    def call(self, inputs, states):
      if len(inputs.shape) is 1:
        inputs = tf.expand_dims(inputs, 0)
      if states is None:
        initial_states = self.get_initial_state(inputs)
      else:
        initial_states = states[0] # (batch, hidden)
      # Update gate, info to pass on (same as simple RNN up until sigmoid)
      #z_t = \sigma(W^zx_t + U^(z)h_{t-1})
      z_in_x = tf.matmul(inputs, self.w_z) # (batch, hidden)
      #assert z_in_x.shape == tf.TensorShape((batch, hidden))
      z_in_h = tf.matmul(initial_states, self.h_z) # (batch, hidden)
      z_t = tf.keras.activations.sigmoid(z_in_x + z_in_h)
      # Reset Gate, how much to forget
      #r_t = \sigma(W^rx_t + U^rh_{t-1})
      r_in_x = tf.matmul(inputs, self.w_r) # (batch, hidden)
      r_in_h = tf.matmul(initial_states, self.h_r) # (batch, hidden)
      r_t = tf.keras.activations.sigmoid(r_in_x + r_in_h)
      # Current memory context
      # hc_t=tanh(Wx_t + r_t . Uh_{t-1})
      hc_in_x = tf.matmul(inputs, self.w_mc) # (batch, hidden)
      hc_in_h = tf.matmul(initial_states, self.h_mc) # (batch, hidden)
      hc_reset = tf.multiply(r_t, hc_in_h) # (batch, hidden)
      h_ct = tf.keras.activations.sigmoid(hc_in_x + hc_reset)

      # Final memory, Combination of current context and previous memories
      # h_t=z_t.h_{t-1} + (1-z_t).hc_t
      # Does order matter here?...
      up_gate_hidden = tf.multiply(z_t, initial_states) # (batch, hidden)
      one_matrix = tf.ones(z_t.shape)
      flip_z = one_matrix - z_t # (batch, hidden)
      # flip * h_ct
      rh_ht = tf.multiply(flip_z, h_ct)
      next_state = up_gate_hidden + rh_ht # (batch, hidden)

      output = tf.matmul(next_state, self.w_y) # (batch, output_size)
      return output, next_state

class MyCrossEntropyLoss(tf.keras.losses.Loss):
  def call(self, y_true, y_pred):
    # -sum([p[i] * log2(q[i]) for i in range len(p)])
    cce = -sum([y_true[i] * log2(y_pred[i]) for i in range(len(y_true))])
    return cce