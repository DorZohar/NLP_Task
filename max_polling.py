from keras import backend as K
from keras.engine.topology import Layer, Input
from keras.layers import Input, Dense, Embedding, LSTM, Lambda, concatenate
from keras.models import Model
import tensorflow as tf
import config as cfg

#input - sentence
#output - k vectors of tuples, (index of word, sentence)
def create_k_max_pooling_model(k):
    input_layer = Input(shape=(cfg.num_of_words + 1,))
    LSTM_hidden_size = 10 #??
    output_layer = LSTM(LSTM_hidden_size, input_shape=(cfg.embedding_size, 1))(input_layer)
    values, indices = tf.nn.top_k(output_layer, k)
    return Model(inputs=input_layer, outputs=[(input_layer, i) for i in indices])

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
