import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import argparse
import json


class SampleSentimentCallback(Callback):
    def __init__(self, sample_text, analyzer):
        super().__init__()
        self.sample_text = sample_text
        self.analyzer = analyzer

    def on_epoch_end(self, epoch, logs=None):
        print("\nSample text sentiment distribution at the end of epoch {}:".format(epoch + 1))
        sentiment_distribution = self.analyzer.predict_sentiment(self.sample_text)
        print(sentiment_distribution)


class SentimentAnalyzer():
    def __init__(self, retrain=False,
                 data_path="../../data/sentiment-data/train.txt",
                 model_path="sentiment.keras", tokenizer_path="tokenizer.json",
                 labels_path="sentiment_labels.json",
                 sample_text="She didn't come today because she lost her dog yesterday!"):
        self.data_path = data_path
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.labels_path = labels_path
        self.data = pd.read_csv(data_path, sep=';')
        self.data.columns = ["Text", "Emotions"]
        self.label_encoder = LabelEncoder()
        self.model = None
        self.max_length = None
        self.sentiment_classes = None
        self.retrain = retrain
        self.sample_text = sample_text

        if self.retrain:
            self.tokenizer = Tokenizer()
        else:
            self.load_tokenizer()

        self.load_or_train_model()

    def save_tokenizer(self):
        tokenizer_json = self.tokenizer.to_json()
        with open(self.tokenizer_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    def load_tokenizer(self):
        with open(self.tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_json = json.load(f)
            self.tokenizer = tokenizer_from_json(tokenizer_json)

    def save_labels(self):
        with open(self.labels_path, 'w') as file:
            json.dump(self.label_encoder.classes_.tolist(), file)

    def load_labels(self):
        with open(self.labels_path, 'r') as file:
            classes = json.load(file)
            self.label_encoder.classes_ = np.array(classes)
            self.sentiment_classes = classes

    def preprocess_data(self):
        texts = self.data["Text"].tolist()
        labels = self.data["Emotions"].tolist()
        
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        self.max_length = max([len(seq) for seq in sequences])
        
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length)
        labels = self.label_encoder.fit_transform(labels)
        one_hot_labels = to_categorical(labels)

        self.sentiment_classes = self.label_encoder.classes_
        return train_test_split(padded_sequences, one_hot_labels, test_size=0.2, stratify=labels)

    def build_model(self, input_dim, output_dim, input_length, units, num_classes):
        optimizer = Adam(learning_rate=0.0005) 
        self.model = Sequential()
        self.model.add(Embedding(input_dim=input_dim, output_dim=output_dim))
        self.model.add(Dropout(0.3))
        self.model.add(Bidirectional(LSTM(units=units)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(units=num_classes, activation='softmax'))
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    def train_model(self, xtrain, ytrain, xtest, ytest, epochs=5, batch_size=32):
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)
        sample_callback = SampleSentimentCallback(self.sample_text, self)

        model_checkpoint = ModelCheckpoint(
            self.model_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )
        self.save_tokenizer()

        return self.model.fit(xtrain, ytrain, epochs=epochs, batch_size=batch_size,
                              validation_data=(xtest, ytest),
                              callbacks=[reduce_lr, sample_callback, model_checkpoint])

    def save_model(self):
        self.model.save(self.model_path)

    def load_or_train_model(self):
        if os.path.exists(self.model_path) and not self.retrain:
            self.model = load_model(self.model_path)
            self.load_labels() 
            print("Loaded existing model.")
        else:
            xtrain, xtest, ytrain, ytest = self.preprocess_data()
            self.build_model(input_dim=len(self.tokenizer.word_index) + 1, output_dim=132, input_length=self.max_length, units=64, num_classes=len(ytrain[0]))
            self.train_model(xtrain, ytrain, xtest, ytest)
            self.save_model()
            self.save_labels()
            print("Trained and saved a new model.")

    def predict_single_sentiment(self, text):
        input_sequence = self.tokenizer.texts_to_sequences([text])
        padded_input_sequence = pad_sequences(input_sequence, maxlen=self.max_length)
        prediction = self.model.predict(padded_input_sequence)
        predicted_label = self.label_encoder.inverse_transform([np.argmax(prediction[0])])
        return predicted_label

    def predict_sentiment(self, text):
        input_sequence = self.tokenizer.texts_to_sequences([text])
        if not input_sequence[0]: # Uniform distribution if no known words
            print("No known words in the input text.")
            return {label: 1/len(self.sentiment_classes) for label in self.sentiment_classes} 

        padded_input_sequence = pad_sequences(input_sequence, maxlen=self.max_length)
        prediction = self.model.predict(padded_input_sequence)
        sentiment_distribution = dict(zip(self.sentiment_classes, prediction[0]))
        return sentiment_distribution


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or load a sentiment analysis model.")
    parser.add_argument("--retrain", action="store_true", help="Retrain the model.")
    args = parser.parse_args()

    sa = SentimentAnalyzer(retrain=args.retrain)

    # Load tokenizer here if not retraining
    if not args.retrain:
        with open('tokenizer.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            sa.tokenizer = tokenizer_from_json(data)

    # Example of making a prediction
    print(sa.predict_sentiment("She didn't come today because she lost her dog yesterday!"))
