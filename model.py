

# Load word2vec embeddings to memory
def load_word2vec():
    raise NotImplementedError


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




