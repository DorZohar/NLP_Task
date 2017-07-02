

# Load word2vec embeddings to memory
from gensim.models import Word2Vec
import numpy as np
import config as cfg


def create_word2vec_embeddings(top_words):

    w2v = Word2Vec.load('C:\\Wiki\\wiki.word2vec.model')
    word_vectors = w2v.wv
    del w2v

    weights = np.zeros((cfg.num_of_words + 1, cfg.embedding_size))

    count = 0

    for i in range(len(top_words)):
        if top_words[i] in word_vectors:
            weights[i] = word_vectors[top_words[i]]
        else:
            count += 1

    cfg.debug_print("Number of words with no representations: %d" % count, 1)

    return weights


# Function that yields a batch for training in every call
def train_generator():
    raise NotImplementedError


# Function that yields a batch for validation in every call
def val_generator():
    raise NotImplementedError


# Reading the data, cleaning and tokenizing (Transforming the text for the mission)
# Can be called from the generators
def preprocess():
    raise NotImplementedError


# Build the main model and compile
def compile_model():
    raise NotImplementedError


# Train the model
def train_model(model):
    raise NotImplementedError




