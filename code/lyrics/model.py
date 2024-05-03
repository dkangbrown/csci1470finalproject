from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional, Embedding
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
import numpy as np
import logging


def generator(sentence_list, next_word_list, word_indices, batch_size):
    index = 0
    while True:
        x = np.zeros((batch_size, len(sentence_list[0])), dtype=np.int32)
        y = np.zeros((batch_size), dtype=np.int32)
        for i in range(batch_size):
            for t, w in enumerate(sentence_list[index % len(sentence_list)]):
                x[i, t] = word_indices.get(w, 0)  # Use 0 if word not found
            y[i] = word_indices.get(next_word_list[index % len(sentence_list)], 0)
            index += 1
        yield x, y


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


class LyricsGenerator:
    def __init__(self, words, model=None):
        self.words = words
        self.word_indices = {w: i for i, w in enumerate(words)}
        self.indices_word = {i: w for i, w in enumerate(words)}
        self.model = model if model else self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=len(self.words), output_dim=1024))
        model.add(Bidirectional(LSTM(128)))
        model.add(Dense(len(self.words)))
        model.add(Activation('softmax'))
        return model

    def train(self, X_train, y_train, X_test, y_test, batch_size):
        logging.info("Training the model...")
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
        callbacks_list = [
            ModelCheckpoint("./checkpoints/LSTM_LYRICS-epoch{epoch:03d}-words%d-sequence%d-minfreq%d-loss{loss:.4f}-acc{accuracy:.4f}-val_loss{val_loss:.4f}-val_acc{val_accuracy:.4f}.keras" %
                            (len(self.words), len(X_train[0]), 7),
                            monitor='val_accuracy', save_best_only=True),
            LambdaCallback(on_epoch_end=self.on_epoch_end),
            EarlyStopping(monitor='val_accuracy', patience=20)
        ]
        self.model.fit(generator(X_train, y_train, self.word_indices, batch_size),
                       steps_per_epoch=int(len(X_train) / batch_size) + 1,
                       verbose=1,
                       epochs=20,
                       callbacks=callbacks_list,
                       validation_data=generator(X_test, y_test, self.word_indices, batch_size),
                       validation_steps=int(len(y_test) / batch_size) + 1)
        logging.info("Model trained successfully.")
        self.model.save('lyrics_generator.h5')
        logging.info("Model saved successfully.")


    def generate_text(self, seed_sentence, diversity=1.0,num_words=50):
        sentence = seed_sentence.split()  # This splits the seed into words, ensuring 'sentence' is a list
        generated_text = []

        for _ in range(num_words):
            x_pred = np.zeros((1, len(sentence)))  # Create input tensor for prediction
            for t, word in enumerate(sentence):
                x_pred[0, t] = self.word_indices.get(word, 0)  # Convert words to indices

            preds = self.model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            
            if next_index in self.indices_word:
                next_word = self.indices_word[next_index]  # Get next word from dictionary
                sentence.append(next_word)  # Append next word as string
                generated_text.append(next_word)
                sentence = sentence[1:]  # Move window forward
            else:
                print(f"Sampled index {next_index} not in dictionary, using fallback word.")
                next_word = self.indices_word[0]  # Use a fallback word if index not found
                sentence.append(next_word)  # Append fallback word
                generated_text.append(next_word)
                sentence = sentence[1:]  # Move window forward

        return ' '.join(generated_text)


    def on_epoch_end(self, epoch, logs):
        # Custom function to be filled based on specific use-case for generating text during training
        print("Generating text after epoch: %d" % epoch)
        seed_text = ['hello', 'from', 'the', 'other', 'side']
        sample_text = self.generate_text(seed_text, num_words=50)
        print("Generated text: ", sample_text)


