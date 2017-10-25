from keras import backend as K
from keras.engine.topology import Layer

from keras.models import Model
import tensorflow as tf
from keras.layers import Input, Dense, Embedding, LSTM, Lambda, Dropout, Flatten, Concatenate, Reshape, \
    Multiply, Bidirectional, RepeatVector, Conv2D, AveragePooling2D

import config as cfg

word_vectors = None

def create_embedding_model(input_layer):
    output_layer = Embedding(cfg.num_of_words,
                             cfg.embedding_size,
                             mask_zero=True,
                             trainable=False,
                             weights=[word_vectors.syn0],
                             input_length=cfg.max_sentence_len)(input_layer)

    return output_layer

def create_2_seq_LSTM_model(sentence1, sentence2, indices_to_remove):

    sentence1 = RepeatVector(cfg.max_sentence_len)(sentence1)

    # Currently not handling batches, as indices are [batch,index,1]
    all_guesses = []
    bi_lstm = Bidirectional(LSTM(cfg.guess_lstm_hidden_layer,
                                 return_sequences=True,
                                 input_shape=(None, cfg.embedding_size),
                                 recurrent_dropout=cfg.guess_lstm_rec_dropout,
                                 dropout=cfg.guess_lstm_input_dropout))
    dense = Dense(cfg.embedding_size)
    for word_index in range(cfg.K_value):

        sentenceMask = 1.0 - tf.expand_dims(tf.one_hot(indices_to_remove[:, word_index], cfg.max_sentence_len), -1)

        # sentencesWithRemovedWords = Lambda(lambda x: tf.map_fn(lambda elems: elems[0] * tf.expand_dims(1 - tf.one_hot(elems[1], cfg.max_sentence_len), -1),
        #                                       [x, indices_to_remove[:, word_index]],
        #                                       dtype=tf.float32))(sentence2)

        sentencesWithRemovedWords = Lambda(lambda x: x * sentenceMask)(sentence2)
        sentencesWithRemovedWords = Concatenate()([sentencesWithRemovedWords, sentence1])

        lstm_outputs = bi_lstm(sentencesWithRemovedWords)
        curGuess = Lambda(lambda x: tf.map_fn(lambda elems: elems[0][elems[1]],
                                              [x, indices_to_remove[:, word_index]],
                                              dtype=tf.float32))(lstm_outputs)

        curGuess = dense(curGuess)
        curGuess = Lambda(lambda x: K.expand_dims(x, -2))(curGuess)

        all_guesses += [curGuess]
    #all_guesses = Lambda(lambda x: guess_single_missing_word([sentence1, sentence2, x]))(indices_to_remove)

    layer_output = Concatenate(dtype='float32', axis=-2)(all_guesses)

    return layer_output


def create_classifier_model(guess, kmax_pooling):

    guess = Reshape((cfg.K_value, 1, cfg.embedding_size))(guess)
    kmax_pooling = Reshape((cfg.K_value, 1, cfg.embedding_size))(kmax_pooling)

    con = Concatenate(axis=1)([guess, kmax_pooling])
    con = Reshape((cfg.K_value, 2 * cfg.embedding_size, 1))(con)

    convLayer = Conv2D(filters=cfg.filters,
                       kernel_size=(1, 2*cfg.embedding_size),
                       activation=cfg.activation,
                       data_format='channels_last')(con)

    avgPooling = AveragePooling2D(pool_size=(cfg.K_value, 1))(convLayer)

    avgPooling = Reshape((cfg.filters, ))(avgPooling)

    dense1 = Dense(cfg.denseSize, input_shape=(cfg.filters,), activation='tanh', name='dense1')(avgPooling)
    dropout = Dropout(cfg.dropoutRate, name='dropout')(dense1)
    output = Dense(cfg.numClasses, activation='softmax', name='output')(dropout)

    return output
