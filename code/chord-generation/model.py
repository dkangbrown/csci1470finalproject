import os
import numpy as np
import pickle
from tqdm import tqdm
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, TimeDistributed, Bidirectional
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import Metric, Precision, Recall
from tensorflow.keras.utils import to_categorical
from config import *

@tf.keras.utils.register_keras_serializable()
class F1Score(Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

    def get_config(self):
        config = super(F1Score, self).get_config()
        return config

def load_corpus(corpus_path):
    """Load and return the data corpus from a pickle file."""
    with open(corpus_path, "rb") as file:
        return pickle.load(file)

def split_data(corpus, val_ratio):
    """Split the corpus into training and validation datasets based on the validation ratio."""
    train_input, train_output = [], []
    val_input, val_output = [], []
    for melody, chords in tqdm(corpus, desc="Splitting data"):
        dataset = (train_input, train_output) if np.random.rand() > val_ratio else (val_input, val_output)
        for i in range(len(melody) - 3):
            dataset[0].append(np.array(melody[i:i+4], dtype=np.float32))  # Convert melody to float32
            dataset[1].append(chords[i:i+4])
    train_input = np.nan_to_num(train_input)
    train_output = np.nan_to_num(train_output)
    return (train_input, to_categorical(np.array(train_output, dtype=np.int32), num_classes=25)), (val_input, to_categorical(np.array(val_output, dtype=np.int32), num_classes=25))


def build_model(input_shape=(4, 12), rnn_size=RNN_SIZE, num_layers=NUM_LAYERS, weights_path=None):
    """Construct and compile a LSTM model with specified parameters."""
    if weights_path and os.path.exists(weights_path):
        model = tf.keras.models.load_model(weights_path)
        return model

    inputs = Input(shape=input_shape, name='input_melody')
    x = TimeDistributed(Dense(12))(inputs)
    for i in range(num_layers):
        x = Bidirectional(LSTM(rnn_size, return_sequences=True, name=f'bidir_lstm_{i+1}'))(x)
        x = Dropout(0.2)(x)
    outputs = TimeDistributed(Dense(25, activation='softmax'))(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', F1Score()])
    return model

def train_model(model, data, data_val, batch_size=BATCH_SIZE, epochs=EPOCHS):
    """Train the model using the provided data and parameters."""
    callbacks = [ModelCheckpoint(filepath=WEIGHTS_PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='min')]
    # Ensure data is in the correct format
    train_x = np.array(data[0], dtype=np.float32)
    train_y = np.array(data[1], dtype=np.float32)
    val_x = np.array(data_val[0], dtype=np.float32)
    val_y = np.array(data_val[1], dtype=np.float32)
    history = model.fit(x=train_x, y=train_y, validation_data=(val_x, val_y), batch_size=batch_size, epochs=epochs, callbacks=callbacks)
    return history

if __name__ == "__main__":
    corpus = load_corpus(CORPUS_PATH)
    data, data_val = split_data(corpus, VAL_RATIO)
    model = build_model()
    history = train_model(model, data, data_val)

