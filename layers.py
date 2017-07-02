from keras import backend as K
from keras.engine.topology import Layer

from keras.models import Model
from keras.layers import Input, Dense, Embedding

import config as cfg


def create_embedding_model(weights):

    input_layer = Input(shape=(cfg.num_of_words + 1,))
    output_layer = Embedding(cfg.num_of_words + 1,
                             cfg.embedding_size,
                             mask_zero = True,
                             trainable = False,
                             weights = [weights],
                             input_length=cfg.max_sentence_len)

    model = Model(input_layer, output_layer)

    return model


def create_k_max_pooling_model():
    raise NotImplementedError


def create_2_seq_LSTM_model():
    raise NotImplementedError


def create_classifier_model():
    raise NotImplementedError




class K_max_pooling(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(K_max_pooling, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(K_max_pooling, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


