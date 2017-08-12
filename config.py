
# Top N words to keep embeddings for
num_of_words = 10000

# Size of the embedding vector
embedding_size = 300

# Batch size
batch_size = 100

# Path of the trained word2vec
word2vec_path = 'C:\\Wiki\\wiki.word2vec.model'

# Max size for a sentence (Larger sentences are trimmed, shorter are padded)
max_sentence_len = 100

# Only debug prints with level of at least debug_level will be printed
debug_level = 1

# Number of epochs for the training step
epochs = 1


def debug_print(string, level):
    if level > debug_level:
        print(string)
