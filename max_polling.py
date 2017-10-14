from keras import backend as K
from keras.engine.topology import Layer, Input
from keras.layers import Input, Dense, Embedding, LSTM, Lambda, concatenate, Activation, TimeDistributed
from keras.models import Model
import tensorflow as tf
import config as cfg


class K_max_pooling(Layer):
    def __init__(self, **kwargs):
        super(K_max_pooling, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(K_max_pooling, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        vals, idxs = tf.nn.top_k(x, cfg.K_value)
        vals = tf.expand_dims(vals, 0)
        idxs = tf.expand_dims(idxs, 0)
        return K.concatenate([vals, K.cast(idxs, 'float32')], axis=0)
        #return idxs

    def compute_output_shape(self, input_shape):
        return (cfg.K_value, 2)


#input - sentence
#output - k vectors of tuples, (index of word, sentence)
def create_k_max_pooling_model(input_layer):
    output_layer = LSTM(1, # cfg.kmax_lstm_hidden_layer,
                        input_shape=(cfg.max_sentence_len, cfg.embedding_size),
                        return_sequences=True)(input_layer)

    #scores = TimeDistributed(Dense(1, activation='tanh'))(output_layer)

    scores = output_layer

    scores = Lambda(lambda x: tf.squeeze(x, -1))(scores)

    print(scores)
    indices_and_vals = K_max_pooling()(scores)
    print(indices_and_vals)

    indices = tf.cast(indices_and_vals[1], 'int32')
    #values = indices_and_vals[0]

    values = Lambda(lambda x: tf.map_fn(lambda elems: K.gather(elems[0], elems[1]), [x, indices], dtype=tf.float32))(input_layer)

    return indices, values

