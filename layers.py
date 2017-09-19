from keras import backend as K
from keras.engine.topology import Layer

from keras.models import Model
from keras.layers import Input, Dense, Embedding, LSTM, Lambda, concatenate, Dropout, Flatten, Concatenate, Reshape, \
    Multiply

import config as cfg

word_vectors = None

def create_embedding_model():
    input_layer = Input(shape=(cfg.num_of_words,))
    output_layer = Embedding(cfg.num_of_words,
                             cfg.embedding_size,
                             mask_zero=True,
                             trainable=False,
                             weights=[word_vectors.syn0],
                             input_length=cfg.max_sentence_len)(input_layer)

    model = Model(input_layer, output_layer)

    return model

def create_2_seq_LSTM_model():
    guess_single_missing_word = guess_missing_word_model()
    sentence1 = Input(shape=(cfg.lstm1_hidden_layer, ), dtype='float32')
    sentence2 = Input(shape=(cfg.max_sentence_len, cfg.embedding_size), dtype='float32')
    indices_to_remove = Input(shape=(cfg.K_value,), dtype='int32')

    # Currently not handling batches, as indices are [batch,index,1]
    all_guesses = []
    for word_index in range(cfg.K_value):
        curIndexToRemove = Lambda(lambda x: x[word_index])(indices_to_remove)
        curGuess = guess_single_missing_word([sentence1, sentence2, curIndexToRemove])
        curGuess = Lambda(lambda x: K.expand_dims(x, -2))(curGuess)
        all_guesses += [curGuess]
    #all_guesses = Lambda(lambda x: guess_single_missing_word([sentence1, sentence2, x]))(indices_to_remove)

    layer_output = Concatenate(dtype='float32', axis=-2)(all_guesses)
    guess_missing_words = Model(inputs=[sentence1, sentence2, indices_to_remove], outputs=layer_output)
    return guess_missing_words


def guess_missing_word_model():
    # Split sentence2 to 2 parts without the word to guess
    sentence2_embedding = Input(shape=(cfg.max_sentence_len, cfg.embedding_size))
    sentence2_index_to_remove = Input(shape=(1,), dtype='int32')

    first_part = Lambda(lambda x: x[:sentence2_index_to_remove[0][0]])(sentence2_embedding)
    second_part = Lambda(lambda x: x[sentence2_index_to_remove[0][0] + 1:])(sentence2_embedding)
    sentence2_formatter = Model(inputs=[sentence2_embedding, sentence2_index_to_remove],
                                outputs=[first_part, second_part])

    # Encode each of the 2 parts of sentence2 (before and after removed word)
    LSTM_hidden_size = cfg.guess_lstm_hidden_layer

    # first_part_model_input = Input(shape=(cfg.embedding_size, 1))
    # first_part_model_output = LSTM(LSTM_hidden_size, input_shape=(cfg.embedding_size, 1))(first_part_model_input)
    # first_part_model = Model(inputs=first_part_model_input, outputs=first_part_model_output)

    # second_part_model_input = Input(shape=(cfg.embedding_size, 1))
    # second_part_model_output = LSTM(LSTM_hidden_size, input_shape=(cfg.embedding_size, 1), go_backwards=True)(
    #     second_part_model_input)
    # second_part_model = Model(inputs=second_part_model_input, outputs=second_part_model_output)

    # Guess single missing word model
    sentence1 = Input(shape=(cfg.lstm1_hidden_layer, ))
    sentence2 = Input(shape=(cfg.max_sentence_len, cfg.embedding_size))
    index_to_remove = Input(shape=(1,))

    (sentence2_first_part, sentence2_second_part) = sentence2_formatter([sentence2, index_to_remove])

    encoded_first_part = LSTM(LSTM_hidden_size, input_shape=(None, cfg.embedding_size))(sentence2_first_part)
    encoded_second_part = LSTM(LSTM_hidden_size, input_shape=(None, cfg.embedding_size), go_backwards=True)(sentence2_second_part)

    combined_embeddings = concatenate([sentence1, encoded_first_part, encoded_second_part], axis=1)
    guess_output = Dense(cfg.num_of_words)(combined_embeddings)

    lamb = Lambda(K.argmax)(guess_output)

    embedding = create_embedding_model()(lamb)

    guess_missing_word = Model(inputs=[sentence1, sentence2, index_to_remove], outputs=embedding)

    return guess_missing_word


def create_classifier_model():

    guess = Input(shape=(cfg.K_value, cfg.embedding_size), dtype='float32')
    kmax_pooling = Input(shape=(cfg.K_value, cfg.embedding_size), dtype='float32')

    mult = Multiply()([guess, kmax_pooling])
    dot = Lambda(lambda x: K.sum(x, axis=-1))(mult)

    dense1 = Dense(cfg.denseSize, input_shape=(cfg.K_value,), activation='tanh', name='dense1')(dot)
    dropout = Dropout(cfg.dropoutRate, name='dropout')(dense1)
    output = Dense(cfg.numClasses, activation='softmax', name='output')(dropout)

    model = Model([guess, kmax_pooling], output)

    return model
