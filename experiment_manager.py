import configuration_store as ConfigurationStore
import os

CONFIG_FILENAME = 'config.py'
MODEL_FILENAME = 'model.h5py'
SCORE_FILENAME = 'score.txt'
TRAINEDMODELS_DIRNAME = 'TrainedModels'

if not os.path.isdir(TRAINEDMODELS_DIRNAME):
	os.makedirs(TRAINEDMODELS_DIRNAME)

def saveExperimentResults(index, configurationFilename, modelFilename, score):
    outputDirectory = os.path.join(TRAINEDMODELS_DIRNAME, 'Experiment%d' % index)
    if not os.path.isdir(outputDirectory):
        os.makedirs(outputDirectory)

    with open(configurationFilename, 'r') as inputFile, open(os.path.join(outputDirectory, CONFIG_FILENAME), 'w') as outputFile:
        outputFile.write(inputFile.read())

    with open(modelFilename, 'rb') as inputFile, open(os.path.join(outputDirectory, MODEL_FILENAME), 'wb') as outputFile:
        outputFile.write(inputFile.read())

    with open(os.path.join(outputDirectory, SCORE_FILENAME), 'w')as outputFile:
        outputFile.write("%f\n" % score)


if __name__ == '__main__':
    experimentIndex = 0
    for configuration in ConfigurationStore.getTestConfigurationFiles():
        experimentIndex += 1

        with open(CONFIG_FILENAME,'w') as configFile:
            configFile.write(configuration)

        import model as ModelTrainer

        score = ModelTrainer.runModel(MODEL_FILENAME)

        saveExperimentResults(experimentIndex, CONFIG_FILENAME, MODEL_FILENAME, score)

        if os.path.isfile(CONFIG_FILENAME):
            os.remove(CONFIG_FILENAME)

        if os.path.isfile(MODEL_FILENAME):
            os.remove(MODEL_FILENAME)

        del ModelTrainer

