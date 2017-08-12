
from gensim.models import Word2Vec
from nltk.tokenize import ToktokTokenizer
import numpy as np
import config as cfg
from keras.preprocessing.sequence import pad_sequences
import keras

word_vectors = {}
label_dict = {
    'neutral': 0,
    'entailment': 1,
    'contradiction': 2
}


tok = ToktokTokenizer()
total_tokens = 0
known_tokens = 0


def load_word2vec():
    global word_vectors

    w2v = Word2Vec.load(cfg.word2vec_path)
    word_vectors = w2v.wv
    del w2v


def process_line(line):
    global total_tokens, known_tokens

    tokenized = tok.tokenize(line)
    total_tokens += len(tokenized)

    w2v_sequence = [word_vectors[t] for t in tokenized if t in word_vectors]
    known_tokens += len(w2v_sequence)

    return w2v_sequence


# Function that yields a batch for training in every call
def generator(path):
    sent1_batch = []
    sent2_batch = []
    labels_batch = []
    size = 0
    while True:
        with open(path, 'r') as file:
            file.readline()
            for line in file:
                text_label, _, _, _, _, sent1, sent2, _, _, _, _, _, _, _, _ = line.split('\t')
                if text_label not in label_dict.keys():
                    continue
                label = label_dict[text_label]
                line1 = process_line(sent1)
                line2 = process_line(sent2)

                if len(line1) == 0 or len(line2) == 0:
                    continue

                sent1_batch.append(line1)
                sent2_batch.append(line2)
                labels_batch.append(label)
                size += 1

                if size >= cfg.batch_size:
                    inputs = {}
                    inputs['sentence1'] = pad_sequences(sent1_batch, maxlen=cfg.max_sentence_len, truncating='post')
                    inputs['sentence2'] = pad_sequences(sent2_batch, maxlen=cfg.max_sentence_len, truncating='post')
                    output = np.zeros((len(labels_batch), len(label_dict.keys())))
                    for i in range(len(labels_batch)):
                        output[i][labels_batch[i]] = 1
                    yield inputs, output
                    sent1_batch = []
                    sent2_batch = []
                    labels_batch = []


# Reading the data, cleaning and tokenizing (Transforming the text for the mission)
# Can be called from the generators
def preprocess():
    raise NotImplementedError


# Build the main model and compile
def compile_model():

    input1 = keras.layers.Input(shape=(None, cfg.embedding_size), name='sentence1')
    input2 = keras.layers.Input(shape=(None, cfg.embedding_size), name='sentence2')

    lstm1 = keras.layers.recurrent.LSTM(300, name='lstm1')(input1)
    lstm2 = keras.layers.recurrent.LSTM(300, name='lstm2')(input2)

    concat = keras.layers.concatenate([lstm1, lstm2])

    dense1 = keras.layers.Dense(300, activation='tanh', name='dense1')(concat)
    output = keras.layers.Dense(3, activation='softmax', name='output')(dense1)

    model = keras.models.Model([input1, input2], output)

    model.compile(
        optimizer=keras.optimizers.rmsprop(),
        loss='categorical_crossentropy',
        metrics=['acc'],
    )

    return model


# Train the model
def train_model(model, train_file, valid_file):

    model.fit_generator(
        epochs=cfg.epochs,
        generator=generator(train_file),
        steps_per_epoch=50,
        validation_data=generator(valid_file),
        validation_steps=50,
        verbose=1,
    )

    return model


if __name__ == '__main__':

    load_word2vec()
    model = compile_model()

    train_model(model, 'train.txt', 'valid.txt')
    model.save('model.h5py')

    print('Fraction of known words: %f' % (known_tokens / total_tokens))

    score = model.evaluate_generator(
        generator=generator('test.txt'),
        steps=50,
    )

    print("Model accuracy: %s" % score[1])
