import os
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Dense
from keras.utils import to_categorical

class SentimentAnalyzer:
    def __init__(self,
                 data_path="data/sentiment-data/train.txt",
                 model_path="sentiment.keras"):
        self.data_path = data_path
        self.model_path = model_path
        self.data = pd.read_csv(data_path, sep=';')
        self.data.columns = ["Text", "Emotions"]
        self.tokenizer = Tokenizer()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.max_length = None

        self.load_or_train_model()

    def preprocess_data(self):
        texts = self.data["Text"].tolist()
        labels = self.data["Emotions"].tolist()
        
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        self.max_length = max([len(seq) for seq in sequences])
        
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length)
        labels = self.label_encoder.fit_transform(labels)
        one_hot_labels = to_categorical(labels)
        
        return train_test_split(padded_sequences, one_hot_labels, test_size=0.2)

    def build_model(self, input_dim, output_dim, input_length, units, num_classes):
        self.model = Sequential()
        self.model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))
        self.model.add(LSTM(units=units))
        self.model.add(Dense(units=num_classes, activation='softmax'))
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    def train_model(self, xtrain, ytrain, xtest, ytest, epochs=5, batch_size=32):
        return self.model.fit(xtrain, ytrain, epochs=epochs, batch_size=batch_size, validation_data=(xtest, ytest))

    def save_model(self):
        self.model.save(self.model_path)

    def load_or_train_model(self):
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            print("Loaded existing model.")
        else:
            xtrain, xtest, ytrain, ytest = self.preprocess_data()
            self.build_model(input_dim=len(self.tokenizer.word_index) + 1, output_dim=132, input_length=self.max_length, units=64, num_classes=len(ytrain[0]))
            self.train_model(xtrain, ytrain, xtest, ytest)
            self.save_model()
            print("Trained and saved a new model.")

    def predict_sentiment(self, text):
        input_sequence = self.tokenizer.texts_to_sequences([text])
        padded_input_sequence = pad_sequences(input_sequence, maxlen=self.max_length)
        prediction = self.model.predict(padded_input_sequence)
        predicted_label = self.label_encoder.inverse_transform([np.argmax(prediction[0])])
        return predicted_label

# Usage
sa = SentimentAnalyzer()

# Example of making a prediction
print(sa.predict_sentiment("She didn't come today because she lost her dog yesterday!"))
