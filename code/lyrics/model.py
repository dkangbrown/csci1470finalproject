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
        data = load_data()
        return prepare_data(data)

    def build_model(self):
        model = Sequential([
            Embedding(input_dim=len(self.words), output_dim=1024),
            Bidirectional(LSTM(128)),
            Dropout(0.5),
            Dense(len(self.words), activation='softmax')
        ])
        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(clipvalue=0.5), metrics=[perplexity])
        return model

    def train_model(self, batch_size):
        X_train, X_test, y_train, y_test = train_test_split(self.valid_seqs, self.next_words, test_size=0.2, random_state=42)
        examples_file = open('examples.txt', "w")
        callbacks_list = [
            ModelCheckpoint(self.model_path, monitor='val_loss', save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_perplexity', mode='min', patience=10, verbose=1),
            LambdaCallback(on_epoch_end=lambda epoch, logs: self.on_epoch_end(epoch, logs, self.model, self.word_indices, self.indices_word, X_train, X_test, examples_file))
        ]
        self.model.fit(
            self.generate_batches(X_train, y_train, BATCH_SIZE),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=EPOCHS,
            callbacks=callbacks_list,
            validation_data=self.generate_batches(X_test, y_test, batch_size),
            validation_steps=len(X_test) // batch_size
        )
        examples_file.close()

    def generate_batches(self, sentence_list, next_word_list, batch_size):
        index = 0
        while True:
            x = np.zeros((batch_size, len(sentence_list[0])), dtype=np.int32)
            y = np.zeros(batch_size, dtype=np.int32)
            for i in range(batch_size):
                x[i] = [self.word_indices.get(word, 0) for word in sentence_list[index % len(sentence_list)]]
                y[i] = self.word_indices.get(next_word_list[index % len(next_word_list)], 0)
                index += 1
            yield x, y

    def load_or_initialize_model(self, retrain):
        if not retrain and os.path.exists(self.model_path):
            return load_model(self.model_path, custom_objects={'perplexity': perplexity})
        else:
            model = self.build_model()
            self.train_model(BATCH_SIZE)
            return model

    def on_epoch_end(epoch, logs, model, word_indices, indices_word, X_train, X_test, examples_file, MIN_SEQ=5):
        """Function invoked at end of each epoch. Writes generated text to a file and returns list of lines."""
        examples_file.write('\n----- Generating text after Epoch: %d\n' % epoch)

        seed_index = np.random.randint(len(X_train + X_test))
        seed = (X_train + X_test)[seed_index]
        initial_seed_text = ' '.join(seed)

        all_generated_lines = []

        for diversity in [0.3, 0.4, 0.5, 0.6, 0.7]:
            examples_file.write('----- Diversity: ' + str(diversity) + '\n')
            examples_file.write('----- Generating with seed:\n"' + initial_seed_text + '"\n')

            sentence = seed[:]
            generated_lines = []

            for i in range(50): 
                x_pred = np.zeros((1, MIN_SEQ))
                for t, word in enumerate(sentence):
                    x_pred[0, t] = word_indices.get(word, 0)

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_word = indices_word[next_index]

                sentence.append(next_word)
                sentence = sentence[1:]  # Slide the window over the sentence

                if next_word.endswith('.') or next_word.endswith('!') or next_word.endswith('?'):
                    generated_line = ' '.join(sentence)
                    generated_lines.append(generated_line)
                    sentence = []  # Start a new line

            for line in generated_lines:
                examples_file.write(line + '\n')
            
            all_generated_lines.extend(generated_lines)
            
            examples_file.write('='*80 + '\n')
        
        examples_file.flush()

        return all_generated_lines


    def generate_text(self, seed_text, diversity=1.0, num_words=50):
        """Generate text from a seed using the trained model, returning a list of lines."""
        sentence = seed_text.split()
        generated = []  # This will store the full sequence of words.
        current_line = []  # This will store words for the current line.
        current_line.extend(sentence)  # Start the current line with the seed.

        for _ in range(num_words):
            x_pred = np.zeros((1, len(sentence)))
            for t, word in enumerate(sentence):
                x_pred[0, t] = self.word_indices.get(word, 0)

            preds = self.model.predict(x_pred, verbose=0)[0]
            next_index = temperature_sampling(preds, diversity)
            next_word = self.indices_word.get(next_index, 'unknown')

            sentence.append(next_word)
            current_line.append(next_word)
            sentence = sentence[1:] 

            if next_word.endswith(('.', '!', '?')):
                generated.append(' '.join(current_line).strip())
                current_line = []

        if current_line:
            generated.append(' '.join(current_line).strip())

        return generated
