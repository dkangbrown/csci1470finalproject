import string

MIN_FREQUENCY = 7
MIN_SEQ = 5
BATCH_SIZE = 32
EPOCHS = 10

TRANSLATOR = str.maketrans('', '', string.punctuation)

MODEL_PATH = 'lyrics.keras'
# MODEL_PATH = 'lyrics_generator.h5'
