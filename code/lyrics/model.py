import os
import numpy as np
import logging
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional, Embedding
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from config import *
from dataloader import load_data, prepare_data
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import keras

@keras.saving.register_keras_serializable()
def perplexity(y_true, y_pred):
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    perplexity = K.exp(K.mean(cross_entropy))
    return perplexity


def generate_batches(sentence_list, next_word_list, word_indices, batch_size):
    """Generator function to yield batches of sequences and next words."""
    index = 0
    while True:
        x = np.zeros((batch_size, len(sentence_list[0])), dtype=np.int32)
        y = np.zeros(batch_size, dtype=np.int32)
        for i in range(batch_size):
            x[i] = [word_indices.get(word, 0) for word in sentence_list[index % len(sentence_list)]]
            y[i] = word_indices.get(next_word_list[index % len(next_word_list)], 0)
            index += 1
        yield x, y


def temperature_sampling(preds, temperature=1.0):
    """Sample an index from a probability array, using a specified temperature."""
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


class LyricsGenerator:
    def __init__(self, retrain=False):
        self.model_path = MODEL_PATH
        self.valid_seqs, self.next_words, self.words = self.prepare_data_and_vocab()
        self.word_indices = {w: i for i, w in enumerate(self.words)}
        self.indices_word = {i: w for i, w in enumerate(self.words)}
        self.model = self.load_or_initialize_model(retrain)

    def prepare_data_and_vocab(self):
        """Load and prepare data, setting up vocabulary and sequences."""
        data = load_data()
        valid_seqs, next_words, words = prepare_data(data)
        return valid_seqs, next_words, words

    def build_model(self):
        """Construct the LSTM model with defined architecture."""
        optimizer = Adam(clipvalue=0.5) # Gradient clipping to avoid exploding gradients
        model = Sequential([
            Embedding(input_dim=len(self.words), output_dim=1024),
            Bidirectional(LSTM(128)),
            Dropout(0.5),
            Dense(len(self.words), activation='softmax')
        ])
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=[perplexity])
        return model

    def train_model(self, batch_size):
        """Train the model using the stored valid sequences and next words."""
        X_train, X_test, y_train, y_test = train_test_split(self.valid_seqs, self.next_words, test_size=0.2, random_state=42)
        callbacks_list = [
            ModelCheckpoint(self.model_path, monitor='val_loss', save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_perplexity', mode='min', patience=10, verbose=1),
            LambdaCallback(on_epoch_end=self.on_epoch_end)
        ]
        self.model.fit(
            generate_batches(X_train, y_train, self.word_indices, batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=EPOCHS,
            callbacks=callbacks_list,
            validation_data=generate_batches(X_test, y_test, self.word_indices, batch_size),
            validation_steps=len(X_test) // batch_size
        )
        logging.info(f"Best model saved to {self.model_path}")

    def load_or_initialize_model(self, retrain):
        """Load an existing model or initialize and train a new one."""
        if not retrain and os.path.exists(self.model_path):
            logging.info(f"Loading model from {self.model_path}")
            return load_model(self.model_path)
        else:
            logging.info("No existing model found or retrain requested, building and training a new model.")
            self.model = self.build_model()
            self.train_model(BATCH_SIZE)
            return self.model

    def generate_text(self, seed_text, diversity=1.0, num_words=50):
        """Generate text from a seed using the trained model."""
        sentence = seed_text.split()
        generated = []
        for _ in range(num_words):
            x_pred = np.zeros((1, len(sentence)))
            for t, word in enumerate(sentence):
                x_pred[0, t] = self.word_indices.get(word, 0)
            preds = self.model.predict(x_pred, verbose=0)[0]
            next_index = temperature_sampling(preds, diversity)
            next_word = self.indices_word.get(next_index, 'unknown')
            sentence.append(next_word)
            generated.append(next_word)
        return ' '.join(generated)

    def on_epoch_end(self, epoch, logs):
        # Generate text after the epoch completion.
        seed_text = "Never gonna give you up"
        diversity = 1.0
        num_words = 50

        generated_text = self.generate_text(seed_text, diversity, num_words)
        print(f"\nGenerated text after epoch {epoch+1}: {generated_text}")
