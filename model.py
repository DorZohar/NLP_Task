import threading

from gensim.models import Word2Vec
from nltk.tokenize import ToktokTokenizer
import numpy as np
import config as cfg
from keras.preprocessing.sequence import pad_sequences
from layers import create_2_seq_LSTM_model, create_classifier_model
import layers
from max_polling import create_k_max_pooling_model
import keras

label_dict = {
    'neutral': 0,
    'entailment': 1,
    'contradiction': 2
}


tok = ToktokTokenizer()
total_tokens = 0
known_tokens = 0


def load_word2vec():

    w2v = Word2Vec.load(cfg.word2vec_path)
    layers.word_vectors = w2v.wv

    del w2v


def process_line(line):
    global total_tokens, known_tokens

    tokenized = tok.tokenize(line)
    total_tokens += len(tokenized)

    w2v_sequence = [layers.word_vectors[t] for t in tokenized if t in layers.word_vectors]
    known_tokens += len(w2v_sequence)

    return w2v_sequence


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


# Function that yields a batch for training in every call
@threadsafe_generator
def generator(path):
    print(1)
    sent1_batch = []
    sent2_batch = []
    labels_batch = []
    size = 0
    while True:
        with open(path, 'r', encoding='utf-8') as file:
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
                    print(inputs)
                    print(output)
                    yield inputs, output
                    sent1_batch = []
                    sent2_batch = []
                    labels_batch = []
                    size = 0


# Reading the data, cleaning and tokenizing (Transforming the text for the mission)
# Can be called from the generators
def preprocess():
    raise NotImplementedError


# Build the main model and compile
def compile_model():

    input1 = keras.layers.Input(shape=(cfg.max_sentence_len, cfg.embedding_size), name='sentence1')
    input2 = keras.layers.Input(shape=(cfg.max_sentence_len, cfg.embedding_size), name='sentence2')

    #embedding1 = create_embedding_model(word_vectors)(input1)

    lstm1 = keras.layers.recurrent.LSTM(cfg.lstm1_hidden_layer, name='lstm1')(input1)

    indexes, kmax_pooling = create_k_max_pooling_model()(input2)

    guess = create_2_seq_LSTM_model()([lstm1, input2, indexes])

    output = create_classifier_model()([guess, kmax_pooling])

    model = keras.models.Model([input1, input2], output)

    model.compile(
        optimizer='RMSProp',
        loss='categorical_crossentropy',
        metrics=['acc']
    )

    return model


# Train the model
def train_model(model, train_file, valid_file):

    model.fit_generator(
        epochs=cfg.epochs,
        generator=generator(train_file),
        steps_per_epoch=5000,
        validation_data=generator(valid_file),
        validation_steps=500,
        workers=6,
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
