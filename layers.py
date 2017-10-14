from keras import backend as K
from keras.engine.topology import Layer

from keras.models import Model
import tensorflow as tf
from keras.layers import Input, Dense, Embedding, LSTM, Lambda, concatenate, Dropout, Flatten, Concatenate, Reshape, \
    Multiply, Bidirectional

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

    # Currently not handling batches, as indices are [batch,index,1]
    all_guesses = []
    bi_lstm = Bidirectional(LSTM(cfg.guess_lstm_hidden_layer, return_sequences=True, input_shape=(None, cfg.embedding_size)))
    dense = Dense(cfg.embedding_size)
    for word_index in range(cfg.K_value):

        print(indices_to_remove)

        sentenceMask = 1.0 - tf.expand_dims(tf.one_hot(indices_to_remove[:, word_index], cfg.max_sentence_len), -1)
        print(sentenceMask)

        # sentencesWithRemovedWords = Lambda(lambda x: tf.map_fn(lambda elems: elems[0] * tf.expand_dims(1 - tf.one_hot(elems[1], cfg.max_sentence_len), -1),
        #                                       [x, indices_to_remove[:, word_index]],
        #                                       dtype=tf.float32))(sentence2)

        sentencesWithRemovedWords = Lambda(lambda x: x * sentenceMask)(sentence2)

        lstm_outputs = bi_lstm(sentencesWithRemovedWords)
        curGuess = Lambda(lambda x: tf.map_fn(lambda elems: elems[0][elems[1]],
                                              [x, indices_to_remove[:, word_index]],
                                              dtype=tf.float32))(lstm_outputs)

        # guess_output = Dense(cfg.num_of_words)(curGuess)
        # lamb = Lambda(K.argmax)(guess_output)
        # embedding = create_embedding_model(lamb)
        # curGuess = Lambda(lambda x: K.expand_dims(x, -2))(embedding)

        print(curGuess)
        curGuess = dense(curGuess)
        print(curGuess)
        curGuess = Lambda(lambda x: K.expand_dims(x, -2))(curGuess)

        all_guesses += [curGuess]
    #all_guesses = Lambda(lambda x: guess_single_missing_word([sentence1, sentence2, x]))(indices_to_remove)

    layer_output = Concatenate(dtype='float32', axis=-2)(all_guesses)

    return layer_output


def create_classifier_model(guess, kmax_pooling):

    mult = Multiply()([guess, kmax_pooling])
    dot = Lambda(lambda x: K.sum(x, axis=-1))(mult)

    dense1 = Dense(cfg.denseSize, input_shape=(cfg.K_value,), activation='tanh', name='dense1')(dot)
    dropout = Dropout(cfg.dropoutRate, name='dropout')(dense1)
    output = Dense(cfg.numClasses, activation='softmax', name='output')(dropout)

    return output
