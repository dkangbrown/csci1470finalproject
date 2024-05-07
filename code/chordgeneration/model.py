import pickle
import os
import numpy as np
from tqdm import trange
from keras.layers import Input, LSTM, Dense, Dropout, TimeDistributed, Bidirectional
from keras import Model
from keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.metrics import Metric, Precision, Recall
import tensorflow as tf
from tensorflow.keras.models import Sequential
from config import *
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

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

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

    def get_config(self):
        config = super(F1Score, self).get_config()
        return config


class MusicModelTrainer:
    def __init__(self, corpus_path=CORPUS_PATH, val_ratio=VAL_RATIO, rnn_size=RNN_SIZE, l2_lambda=0.001,
                 num_layers=NUM_LAYERS, batch_size=BATCH_SIZE, epochs=EPOCHS, weights_path=WEIGHTS_PATH):
        self.corpus_path = corpus_path
        self.val_ratio = val_ratio
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.weights_path = weights_path
        self.l2_lambda = l2_lambda

    def create_training_data(self):
        with open(self.corpus_path, "rb") as filepath:
            data_corpus = pickle.load(filepath)

        input_melody = []
        output_chord = []
        input_melody_val = []
        output_chord_val = []

        np.random.seed(0)

        for songs_idx in trange(len(data_corpus)):
            song = data_corpus[songs_idx]
            train_or_val = 'train' if np.random.rand() > self.val_ratio else 'val'
            song_melody = song[0][0]
            song_chord = song[0][1]

            for idx in range(len(song_melody) - 3):
                melody = song_melody[idx:idx + 4]
                chord = song_chord[idx:idx + 4]

                if train_or_val == 'train':
                    input_melody.append(melody)
                    output_chord.append(chord)
                else:
                    input_melody_val.append(melody)
                    output_chord_val.append(chord)

        print("Successfully read %d pieces" % len(data_corpus))
        onehot_chord = to_categorical(output_chord, num_classes=25)
        onehot_chord_val = to_categorical(output_chord_val, num_classes=25) if len(input_melody_val) != 0 else []

        return (input_melody, onehot_chord), (input_melody_val, onehot_chord_val)

    def build_model(self):
        regularizer = l2(self.l2_lambda)
        model = Sequential([
            Input(shape=(4, 12), name='input_melody'),
            TimeDistributed(Dense(12, kernel_regularizer=regularizer)),
            Bidirectional(LSTM(units=self.rnn_size, return_sequences=True, name='melody_1')),
            TimeDistributed(Dense(units=self.rnn_size, activation='tanh', kernel_regularizer=regularizer)),
            Dropout(0.5),
            Bidirectional(LSTM(units=self.rnn_size, return_sequences=True, name='melody_2')),
            TimeDistributed(Dense(units=self.rnn_size, activation='tanh', kernel_regularizer=regularizer)),
            Dropout(0.5),
            TimeDistributed(Dense(25, activation='softmax', kernel_regularizer=regularizer))
        ])       

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[F1Score()])

        model.summary()

        if self.weights_path and os.path.exists(self.weights_path):
            model.load_weights(self.weights_path)

        return model

    def train_model(self, data, data_val):
        model = self.build_model()
        if os.path.exists(self.weights_path):
            try:
                model.load_weights(self.weights_path)
                print("Checkpoint loaded.")
            except:
                os.remove(self.weights_path)
                print("Checkpoint deleted.")

        monitor = 'val_loss' if len(data_val[0]) != 0 else 'loss'

        checkpoint = ModelCheckpoint(filepath=self.weights_path,
                                     monitor=monitor,
                                     verbose=0,
                                     save_best_only=True,
                                     mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        if len(data_val[0]) != 0:
            history = model.fit(x={'input_melody': np.array(data[0])},
                                y=np.array(data[1]),
                                validation_data=({'input_melody': np.array(data_val[0])}, data_val[1]),
                                batch_size=self.batch_size,
                                epochs=self.epochs,
                                verbose=1,
                                callbacks=[checkpoint, early_stopping])
        else:
            history = model.fit(x={'input_melody': np.array(data[0])},
                                y=np.array(data[1]),
                                batch_size=self.batch_size,
                                epochs=self.epochs,
                                verbose=1,
                                callbacks=[checkpoint, early_stopping])

        return history


if __name__ == "__main__":
    trainer = MusicModelTrainer()
    data, data_val = trainer.create_training_data()
    trainer.train_model(data, data_val)
