from keras import backend as K
from keras.engine.topology import Layer

from keras.models import Model
from keras.layers import Input, Dense, Embedding, LSTM, Lambda, concatenate

import config as cfg

def create_embedding_model(weights):
    input_layer = Input(shape=(cfg.num_of_words + 1,))
    output_layer = Embedding(cfg.num_of_words + 1,
                             cfg.embedding_size,
                             mask_zero=True,
                             trainable=False,
                             weights=[weights],
                             input_length=cfg.max_sentence_len)

    model = Model(input_layer, output_layer)

    return model

def create_2_seq_LSTM_model():
    guess_single_missing_word = guess_missing_word_model()
    sentence1 = Input(shape=(cfg.embedding_size, 1), dtype='float32')
    sentence2 = Input(shape=(cfg.embedding_size, 1), dtype='float32')
    indices_to_remove = Input(shape=(cfg.K_value, 1), dtype='int32')

    # Currently not handling batches, as indices are [batch,index,1]
    all_guesses = []
    for word_index in range(cfg.K_value):
        curIndexToRemove = Lambda(lambda x: x[0][word_index][0])(indices_to_remove)
        curGuess = guess_single_missing_word([sentence1, sentence2, curIndexToRemove])
        all_guesses += [curGuess]

    layer_output = all_guesses
    guess_missing_words = Model(inputs=[sentence1, sentence2, indices_to_remove], outputs=layer_output)
    return guess_missing_words


def guess_missing_word_model():
    # Split sentence2 to 2 parts without the word to guess
    sentence2_embedding = Input(shape=(cfg.embedding_size, 1))
    sentence2_index_to_remove = Input(shape=(1, 1), dtype='int32')

    first_part = Lambda(lambda x: x[:sentence2_index_to_remove[0][0][0]])(sentence2_embedding)
    second_part = Lambda(lambda x: x[sentence2_index_to_remove[0][0][0] + 1:])(sentence2_embedding)
    sentence2_formatter = Model(inputs=[sentence2_embedding, sentence2_index_to_remove],
                                outputs=[first_part, second_part])

    # Encode each of the 2 parts of sentence2 (before and after removed word)
    LSTM_hidden_size = 10

    sentence1_model_input = Input(shape=(cfg.embedding_size, 1))
    sentence1_model_output = LSTM(LSTM_hidden_size, input_shape=(cfg.embedding_size, 1))(sentence1_model_input	)
    sentence1_model = Model(inputs=sentence1_model_input, outputs=sentence1_model_output)

    first_part_model_input = Input(shape=(cfg.embedding_size, 1))
    first_part_model_output = LSTM(LSTM_hidden_size, input_shape=(cfg.embedding_size, 1))(first_part_model_input)
    first_part_model = Model(inputs=first_part_model_input, outputs=first_part_model_output)

    second_part_model_input = Input(shape=(cfg.embedding_size, 1))
    second_part_model_output = LSTM(LSTM_hidden_size, input_shape=(cfg.embedding_size, 1), go_backwards=True)(
        second_part_model_input)
    second_part_model = Model(inputs=second_part_model_input, outputs=second_part_model_output)

    # Guess single missing word model
    sentence1 = Input(shape=(cfg.embedding_size, 1))
    sentence2 = Input(shape=(cfg.embedding_size, 1))
    index_to_remove = Input(shape=(1, 1))

    (sentence2_first_part, sentence2_second_part) = sentence2_formatter([sentence2, index_to_remove])

    sentence1_encoding = sentence1_model(sentence1)
    encoded_first_part = first_part_model(sentence2_first_part)
    encoded_second_part = second_part_model(sentence2_second_part)

    combined_embeddings = concatenate([sentence1_encoding, encoded_first_part, encoded_second_part], axis=1)
    guess_output = Dense(cfg.num_of_words)(combined_embeddings)
    guess_missing_word = Model(inputs=[sentence1, sentence2, index_to_remove], outputs=guess_output)

    return guess_missing_word

def create_classifier_model():
    raise NotImplementedError