from keras import backend as K
from keras.engine.topology import Layer, Input
from keras.layers import Input, Dense, Embedding, LSTM, Lambda, concatenate, Activation
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
        vals, idxs = tf.nn.top_k(x, cfg.K_value, name="Blah")
        return K.concatenate([vals, K.cast(idxs, 'float32')], axis=0)

    def compute_output_shape(self, input_shape):
        return (cfg.K_value, 2)


#input - sentence
#output - k vectors of tuples, (index of word, sentence)
def create_k_max_pooling_model():
    input_layer = Input(shape=(cfg.max_sentence_len, cfg.embedding_size))
    output_layer = LSTM(cfg.kmax_lstm_hidden_layer,
                        input_shape=(cfg.max_sentence_len, cfg.embedding_size),
                        return_sequences=True)(input_layer)
    values_indices = K_max_pooling()(output_layer)
    indices = Lambda(lambda x: x[:, 1])(values_indices)
    indices = Lambda(lambda x: K.cast(x, 'int32'), dtype='int32')(indices)

    values = Lambda(lambda x: tf.map_fn(lambda elems: K.gather(elems[0], K.cast(elems[1], 'int32')), [x, indices], dtype=tf.float32))(input_layer)

    return Model(inputs=input_layer, outputs=[indices, values])

