from collections import namedtuple

CommonConfiguration=namedtuple('CommonConfiguration',[
    'num_of_words', # Top N words to keep embeddings for
    'embedding_size', # Size of the embedding vector
    'batch_size', # Batch size
    'word2vec_path', # Path of the trained word2vec
    'max_sentence_len', # Max size for a sentence (Larger sentences are trimmed, shorter are padded)
    'debug_level', # Only debug prints with level of at least debug_level will be printed
    'epochs', # Number of epochs for the training step
    'numClasses',
    'modelType',
    'model_path' # Path of the final model
])

ModelConfiguration=namedtuple('ModelConfiguration',[
    'K_value', # Number of outputs of K-max layer (number of chosen values)
    'filters',
    'kmax_lstm_hidden_layer','kmax_lstm_rec_dropout','kmax_lstm_input_dropout',
    'guess_lstm_hidden_layer','guess_lstm_rec_dropout','guess_lstm_input_dropout',
    'lstm1_hidden_layer','lstm1_rec_dropout','lstm1_input_dropout',
    'denseSize',
    'dropoutRate','learnRate','activation'])

commonModelConfiguration = CommonConfiguration(
    num_of_words=572123, embedding_size = 300, batch_size = 100, word2vec_path = "'wiki.word2vec.bin'",
    max_sentence_len = 30, debug_level = 1, epochs = 20, numClasses = 3, modelType = 1,
    model_path = "'Models/model1_{epoch:02d}-{val_acc:.2f}.hdf5'"
 )

modelConfigurationsList = [
    ModelConfiguration(K_value=8, filters = 300,
                       kmax_lstm_hidden_layer = 10, kmax_lstm_rec_dropout = 0.1, kmax_lstm_input_dropout = 0.25,
                       guess_lstm_hidden_layer = 150, guess_lstm_rec_dropout = 0.1, guess_lstm_input_dropout = 0.25,
                       lstm1_hidden_layer = 300, lstm1_rec_dropout = 0.1, lstm1_input_dropout = 0.25,
                       denseSize = 300,
                       dropoutRate = 0.25, learnRate = 0.0005, activation = "'tanh'"
                       ),
    ModelConfiguration(K_value=8, filters = 100,
                       kmax_lstm_hidden_layer = 10, kmax_lstm_rec_dropout = 0.1, kmax_lstm_input_dropout = 0.25,
                       guess_lstm_hidden_layer = 150, guess_lstm_rec_dropout = 0.1, guess_lstm_input_dropout = 0.25,
                       lstm1_hidden_layer = 300, lstm1_rec_dropout = 0.1, lstm1_input_dropout = 0.25,
                       denseSize = 300,
                       dropoutRate = 0.25, learnRate = 0.001, activation = "'tanh'"
                       ),
    ModelConfiguration(K_value=8, filters = 500,
                       kmax_lstm_hidden_layer = 10, kmax_lstm_rec_dropout = 0.15, kmax_lstm_input_dropout = 0.3,
                       guess_lstm_hidden_layer = 150, guess_lstm_rec_dropout = 0.15, guess_lstm_input_dropout = 0.3,
                       lstm1_hidden_layer = 300, lstm1_rec_dropout = 0.15, lstm1_input_dropout = 0.3,
                       denseSize = 300,
                       dropoutRate = 0.3, learnRate = 0.0005, activation = "'tanh'"
                       ),
    ModelConfiguration(K_value=8, filters = 300,
                       kmax_lstm_hidden_layer = 10, kmax_lstm_rec_dropout = 0.1, kmax_lstm_input_dropout = 0.25,
                       guess_lstm_hidden_layer = 150, guess_lstm_rec_dropout = 0.1, guess_lstm_input_dropout = 0.25,
                       lstm1_hidden_layer = 500, lstm1_rec_dropout = 0.1, lstm1_input_dropout = 0.25,
                       denseSize = 500,
                       dropoutRate = 0.25, learnRate = 0.0005, activation = "'tanh'"
                       ),
    ModelConfiguration(K_value=8, filters = 300,
                       kmax_lstm_hidden_layer = 10, kmax_lstm_rec_dropout = 0.1, kmax_lstm_input_dropout = 0.25,
                       guess_lstm_hidden_layer = 150, guess_lstm_rec_dropout = 0.1, guess_lstm_input_dropout = 0.25,
                       lstm1_hidden_layer = 300, lstm1_rec_dropout = 0.1, lstm1_input_dropout = 0.25,
                       denseSize = 300,
                       dropoutRate = 0.25, learnRate = 0.01, activation = "'tanh'"
                       ),
    ModelConfiguration(K_value=6, filters = 300,
                       kmax_lstm_hidden_layer = 10, kmax_lstm_rec_dropout = 0.1, kmax_lstm_input_dropout = 0.25,
                       guess_lstm_hidden_layer = 150, guess_lstm_rec_dropout = 0.1, guess_lstm_input_dropout = 0.25,
                       lstm1_hidden_layer = 300, lstm1_rec_dropout = 0.1, lstm1_input_dropout = 0.25,
                       denseSize = 300,
                       dropoutRate = 0.25, learnRate = 0.0005, activation = "'tanh'"
                       ),

]

def getTestConfigurationFiles():
    for modelConf in modelConfigurationsList:
        outputConfigurationFile = generateConfigurationFileContentFromModelConfiguration(modelConf)
        yield outputConfigurationFile

def generateConfigurationFileContentFromModelConfiguration(modelConf):
    outputFileContent = ""
    for commonField in CommonConfiguration._fields:
        outputFileContent += str(commonField) + " = " + str(getattr(commonModelConfiguration,commonField))+"\r\n"

    for modelField in ModelConfiguration._fields:
        outputFileContent += str(modelField) + " = " + str(getattr(modelConf,modelField))+"\r\n"

    outputFileContent += "def debug_print(string, level):"+"\r\n"
    outputFileContent += "  if level > debug_level:"+"\r\n"
    outputFileContent += "    print(string)"+"\r\n"

    return outputFileContent
