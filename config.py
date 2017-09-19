
# Top N words to keep embeddings for
num_of_words = 572123

# Size of the embedding vector
embedding_size = 300

# Batch size
batch_size = 100

# Path of the trained word2vec
word2vec_path = 'C:\\Wiki\\wiki.word2vec.model'

# Max size for a sentence (Larger sentences are trimmed, shorter are padded)
max_sentence_len = 30

# Only debug prints with level of at least debug_level will be printed
debug_level = 1

# Number of epochs for the training step
epochs = 3

# Number of outputs of K-max layer (number of chosen values)
K_value = 5

kmax_lstm_hidden_layer = 10

guess_lstm_hidden_layer = 10

lstm1_hidden_layer = 300

denseSize = 300

dropoutRate = 0.25

numClasses = 3

def debug_print(string, level):
    if level > debug_level:
        print(string)
