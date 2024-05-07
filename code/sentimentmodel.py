import argparse
import os
import re
import string
import tensorflow as tf
import collections
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
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
                 data_path="../data/sentiment-data/train.txt",
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
        return list(prediction[0])
        #sentiment_distribution = dict(zip(self.sentiment_classes, prediction[0]))
        #return sentiment_distribution

class SentimentToChordModel:
    def __init__(self, model_path='text_to_chord_model.keras', embed_size=64, hidden_size=72, batch_size=16, epochs=1, validation_split=0.2):
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.input_vocab_size = None
        self.target_vocab_size = None
        self.model = None
        self.model_path = model_path
        self.input_vocab = None
        self.target_vocab = None
        self.word_count = None
        self.encode_input_lyr = None
        self.encode_embedding_lyr = None
        self.encode_gru_lyr = None
        self.sa = SentimentAnalyzer(retrain=True)

    @staticmethod
    def preprocess_song(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        chord_regex = re.compile(r"^[A-G][#b]?(7|5|M|maj7|maj|M7|mmaj7|min7|m|min|dim|dim7|aug|\+|sus2|sus4|7sus2|7sus4)?(add)?[0-9]*/?[A-G]?[#b]?$")
        section_regex = re.compile(r"^(\(|\[)?[\#]?(chorus|Chorus|CHORUS|verse|Verse|VERSE|intro|Intro|INTRO|outro|Outro|OUTRO|bridge|Bridge|BRIDGE|interlude|Interlude|INTERLUDE|instrumental|Instrumental|INSTRUMENTAL|solo|Solo|SOLO)*( )?[0-9]*(\:|\.)?(\]|\))?$")
        accidental_regex = re.compile(r"^\w[#b]")

        current_section = None
        key_to_pitch = {
            'A': 0, 'A#': 1, 'Bb': 1, 'B': 2, 'Cb': 2, 'B#': 3, 'C': 3, 'C#': 4, 'Db': 4, 'D': 5, 'D#': 6,
            'Eb': 6, 'E': 7, 'Fb': 7, 'E#': 8, 'F': 8, 'F#': 9, 'Gb': 9, 'G': 10, 'G#': 11, 'Ab': 11
        }
        qual_to_num = {
            '': 0, '5': 0, 'M': 0, 'maj7': 1, 'maj': 1, 'M7': 1, '7': 2, 'm': 3, 'mmaj7': 3,
            'min7': 4, 'm7': 4, 'dim': 5, 'dim7': 5, '+': 6, '+5': 6, 'aug': 6, 'sus2': 7,
            '7sus2': 7, '2': 7, 'sus4': 8, '7sus4': 8, '4': 8
        }

        def get_root_pitch(chord):
            if accidental_regex.match(chord):
                root_note = chord[0] + chord[1]
            else:
                root_note = chord[0]
            return key_to_pitch[root_note]

        verses = []
        key = lines[0].strip().split(' ')[1]
        key_pitch = get_root_pitch(key)
        lines = lines[1:]
        versenum = -1
        isbreak = 0

        def get_rel_pitch(pitch):
            if pitch >= key_pitch:
                chord_rel_pitch = pitch - key_pitch
            else:
                chord_rel_pitch = 12 + pitch - key_pitch
            return chord_rel_pitch

        def chord_to_vector(chord):
            chord_rel_pitch = get_rel_pitch(get_root_pitch(chord))
            base_split = chord.split('/')
            if len(base_split) == 2:
                base_rel_pitch = get_rel_pitch(get_root_pitch(base_split[1]))
                chord = base_split[0]
            else:
                base_rel_pitch = chord_rel_pitch

            if accidental_regex.match(chord):
                qual = chord[2:]
            else:
                qual = chord[1:]

            if 'add' in qual:
                qual = qual[0:qual.index('add')]
            elif '6' in qual:
                qual = qual[0:qual.index('6')]
            elif '9' in qual:
                qual = qual[0:qual.index('9')]
            elif '11' in qual:
                qual = qual[0:qual.index('11')]
            elif '13' in qual:
                qual = qual[0:qual.index('13')]
            elif '15' in qual:
                qual = qual[0:qual.index('15')]
            elif '17' in qual:
                qual = qual[0:qual.index('17')]

            return [chord_rel_pitch, qual_to_num[qual]]

        def encode_chord(chord):
            """ Encode chord components into a single integer. """
            return chord[0] * 8 + chord[1]

        def get_chord_list(line):
            line = line.strip()
            chord_list = line.split(" ")
            chord_list = [x for x in chord_list if x]
            return [encode_chord(chord_to_vector(x)) for x in chord_list]

        for line in lines:
            line = line.strip().replace('\'','').replace('�','').replace('|', ' ').replace('(', ' ').replace(')', ' ').replace('-', ' ').replace('%', ' ')\
                .replace('\t', ' ').replace('\\', '/').replace('/ ', ' ').replace('*', ' ').replace('@', ' ')\
                .replace(',  ', '   ').replace('x2', ' ').replace('x3', ' ').replace('x4', ' ').replace('x5', ' ')\
                .replace('x6', ' ').replace('x7', ' ').replace('x8', ' ')
            if section_regex.match(line):
                if isbreak == 0:
                    isbreak = 1
                    versenum += 1
                    verses.append({'text': "", 'chords': []})
            else:
                if isbreak == 1:
                    isbreak = 0
                is_chord_line = True
                for word in line.split():
                    if not(is_chord_line and chord_regex.match(word)):
                        is_chord_line = False
                if is_chord_line:
                    verses[versenum]['chords'] += get_chord_list(line)
                else:
                    verses[versenum]['text'] += line + " "

        return [x for x in verses if x['chords']]

    def prepare_data(self, preprocessed_pairs):
        """ Prepare data for model input from texts and chord vectors. """
        inputs, targets = [], []

        def encode_chord(root_pitch, base_pitch, quality):
            return root_pitch * 8 + quality
        
        # self.word_count = collections.Counter()

        for pair in preprocessed_pairs:
            # split_input = pair.get("text").translate(str.maketrans('', '', string.punctuation)).lower().split()
            # split_input.append("<STOP>")
            # self.word_count.update(split_input)
            inputs.append(pair.get("text"))
            split_targets = pair.get("chords")
            split_targets.append(96)
            targets.append(split_targets)

        #input_data = [list(map(lambda x: sa.predict_sentiment(x), i)) for i in inputs]
        input_data = []
        for d,i in enumerate(inputs):
            input_data.append(self.sa.predict_sentiment(i))
            print(d)
        # def unk_text(texts, minimum_frequency):
        #     for text in texts:
        #         for index, word in enumerate(text):
        #             if self.word_count[word] <= minimum_frequency:
        #                 text[index] = '<unk>'
        # unk_text(inputs,3)

        # unique_input_words = sorted(set([i for j in inputs for i in j]))
        # self.input_vocab = {w: i for i, w in enumerate(unique_input_words)}

        # input_data = [list(map(lambda x: self.input_vocab.get(x), i)) for i in inputs]
        # self.input_vocab_size = len(unique_input_words)

        target_data = targets
        self.target_vocab_size = 97

        return input_data, target_data

    def build_model(self):

        self.encode_input_lyr = tf.keras.Input(shape=(None,))
        self.encode_embedding_lyr = tf.keras.layers.Embedding(6, self.embed_size)
        self.encode_gru_lyr = tf.keras.layers.GRU(self.hidden_size, return_state=True)
        inputs = self.encode_input_lyr
        input_embedding = self.encode_embedding_lyr(inputs)
        encoder_output, state = self.encode_gru_lyr(input_embedding)

        targets = tf.keras.Input(shape=(None,))
        target_embedding = tf.keras.layers.Embedding(self.target_vocab_size, self.embed_size)(targets)
        decoder_output, _ = tf.keras.layers.GRU(self.hidden_size, return_sequences=True, return_state=True)(target_embedding, initial_state=state)
        classifier_logits = tf.keras.layers.Dense(self.target_vocab_size)(decoder_output)

        self.model = tf.keras.Model(inputs=[inputs, targets], outputs=classifier_logits)

    @staticmethod
    def perplexity(logits, labels):
        return tf.exp(tf.reduce_mean(tf.keras.metrics.sparse_categorical_crossentropy(logits, labels), axis=-1))

    def compile_model(self):
        optimizer = tf.keras.optimizers.Adam()
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = [self.perplexity]
        metrics2 = ['accuracy']
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics2)

    def train_model(self, input_data, target_inputs, target_labels):
        self.model.fit(
            x=[input_data, target_inputs],
            y=target_labels,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split
        )

    def save_model(self):
        self.model.save(self.model_path)
    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_path, custom_objects={'perplexity': self.perplexity})

    def run(self, preprocessed_pairs, retrain=False):
        input_data, target_data = self.prepare_data(preprocessed_pairs)
        target_inputs = [i[:-1] for i in target_data]
        target_labels = [i[1:] for i in target_data]

        target_inputs = tf.keras.preprocessing.sequence.pad_sequences(target_inputs, padding='post')
        target_labels = tf.keras.preprocessing.sequence.pad_sequences(target_labels, padding='post')

        if retrain or not os.path.exists(self.model_path):
            self.build_model()
            self.compile_model()
            self.model.summary()
            self.train_model(input_data, target_inputs, target_labels)
            self.save_model()
        else:
            self.load_model()
            print("Loaded existing model.")

    def predict(self, text):
        # split_input = text.translate(str.maketrans('', '', string.punctuation)).lower().split()
        # split_input.append("<STOP>")

        # def unk_text(minimum_frequency):
        #     for index, word in enumerate(split_input):
        #         if self.word_count[word] <= minimum_frequency:
        #             split_input[index] = '<unk>'
        # unk_text(3)

        # print(split_input)

        # input_data = tf.convert_to_tensor(list(map(lambda x: self.input_vocab.get(x), split_input)), dtype=tf.int32)
        # input_data = tf.reshape(input_data, [1, -1])

        # print(input_data)

        input_data = tf.convert_to_tensor(self.sa.predict_sentiment(text))
        input_data = tf.reshape(input_data, [1, -1])

        decoder_input = tf.convert_to_tensor([96], dtype=tf.int32)
        decoder_input = tf.reshape(decoder_input, [1, -1])

        print(input_data)
        output = []

        temp = 0.05

        for _ in range(15):  # Arbitrary output length limit
            prediction_logits = self.model.predict([input_data, decoder_input])
            probs = tf.nn.softmax(prediction_logits / temp).numpy()[0, -1, :]
            attempts = 0
            stop_token = "<STOP>"

            next_token = 96
            # print(probs)
            if output:
                while next_token == 96 or next_token == output[-1]:
                    indices = np.argsort(probs)[:-10]
                    probs[indices] = 0
                    probs = probs / np.sum(probs)
                    next_token = np.random.choice(len(probs), p=probs)
            else:
                while next_token == 96:
                    indices = np.argsort(probs)[:-15]
                    probs[indices] = 0
                    probs = probs / np.sum(probs)
                    next_token = np.random.choice(len(probs), p=probs)
            # predicted_id = tf.argmax(prediction_logits[0, -1, :]).numpy()
            output.append(next_token)
            if next_token == stop_token:
                break
            #decoder_input = tf.concat([decoder_input, [[predicted_id]]], axis=1)
            decoder_input = tf.convert_to_tensor([output], dtype=tf.int32)

        # reverse_target_vocab = {i: w for w, i in self.target_vocab.items()}
        # chords = [reverse_target_vocab.get(i, "<UNK>") for i in output]
        return self.decode_chords(output)

    def decode_chords(self, chords):
        new_chords = []
        key_to_pitch = {
            0: 'A', 1: 'Bb', 2: 'B', 3: 'C', 4: 'C#', 5: 'D', 6: 'Eb', 7: 'E', 8: 'F', 9: 'F#', 10: 'G', 11: 'Ab'
        }
        qual_to_num = {
            0: '', 1: 'maj7', 2: '7', 3: 'm', 4: 'min7', 5: 'dim', 6: 'aug', 7: 'sus2', 8: 'sus4'
        }
        for chord in chords:
            if chord == 96:
                new_chords.append('<unk>')
                continue
            key = key_to_pitch[np.floor(chord / 8)]
            qual = qual_to_num[chord % 8]
            new_chords.append(key + qual)
        
        return new_chords

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or load a text-to-chord generation model.")
    parser.add_argument("--retrain", action="store_true", help="Retrain the model.")
    parser.add_argument("--text", type=str, help="Text to predict chord progression for.")
    args = parser.parse_args()

    Path = "../data/chord-lyric-text/"
    filelist = os.listdir(Path)
    preprocessed_pairs = []
    file_name = re.compile(r"^([A-R]|[a-r])")

    for i in filelist:
        if i.endswith(".txt"):
            song_data = SentimentToChordModel.preprocess_song(Path + i)
            preprocessed_pairs += song_data

    model = SentimentToChordModel()

    model.run(preprocessed_pairs, retrain=args.retrain)
    model.model.summary()

    if args.text:
        print("Predicting chord progression for text: ", args.text)
        print(model.predict(args.text))


# import utils
# import numpy as np
# import tensorflow as tf
# from build import ChordLyricProcessor
#
# # architecture: transformer that takes in text and outputs chords
# # train on lyrics + chords dataset
#
# class Transformer(tf.keras.Model):
#   def __init__(self, *, num_layers, d_model, num_heads, dff,
#                input_vocab_size, target_vocab_size, dropout_rate=0.1):
#     super().__init__()
#     self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
#                            num_heads=num_heads, dff=dff,
#                            vocab_size=input_vocab_size,
#                            dropout_rate=dropout_rate)
#
#     self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
#                            num_heads=num_heads, dff=dff,
#                            vocab_size=target_vocab_size,
#                            dropout_rate=dropout_rate)
#
#     self.final_layer = tf.keras.layers.Dense(target_vocab_size)
#
#   def call(self, inputs):
#     # To use a Keras model with `.fit` you must pass all your inputs in the
#     # first argument.
#     context, x  = inputs
#
#     context = self.encoder(context)  # (batch_size, context_len, d_model)
#
#     x = self.decoder(x, context)  # (batch_size, target_len, d_model)
#
#     # Final linear layer output.
#     logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)
#
#     try:
#       # Drop the keras mask, so it doesn't scale the losses/metrics.
#       # b/250038731
#       del logits._keras_mask
#     except AttributeError:
#       pass
#
#     # Return the final output and the attention weights.
#     return logits
#     
# class PositionalEmbedding(tf.keras.layers.Layer):
#     def __init__(self, vocab_size, d_model):
#         super().__init__()
#         self.d_model = d_model
#         self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
#         self.pos_encoding = positional_encoding(length=2048, depth=d_model)
#
#     def compute_mask(self, *args, **kwargs):
#         return self.embedding.compute_mask(*args, **kwargs)
#
#     def call(self, x):
#         length = tf.shape(x)[1]
#         x = self.embedding(x)
#         # This factor sets the relative scale of the embedding and positonal_encoding.
#         x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
#         x = x + self.pos_encoding[tf.newaxis, :length, :]
#         return x
#
# class BaseAttention(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super().__init__()
#         self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
#         self.layernorm = tf.keras.layers.LayerNormalization()
#         self.add = tf.keras.layers.Add()
#
#
# class CrossAttention(BaseAttention):
#     def call(self, x, context):
#         attn_output, attn_scores = self.mha(
#             query=x,
#             key=context,
#             value=context,
#             return_attention_scores=True)
#
#         # Cache the attention scores for plotting later.
#         self.last_attn_scores = attn_scores
#
#         x = self.add([x, attn_output])
#         x = self.layernorm(x)
#
#         return x
#
# class GlobalSelfAttention(BaseAttention):
#     def call(self, x):
#         attn_output = self.mha(
#             query=x,
#             value=x,
#             key=x)
#         x = self.add([x, attn_output])
#         x = self.layernorm(x)
#         return x
#
# class CausalSelfAttention(BaseAttention):
#     def call(self, x):
#         attn_output = self.mha(
#             query=x,
#             value=x,
#             key=x,
#             use_causal_mask = True)
#         x = self.add([x, attn_output])
#         x = self.layernorm(x)
#         return x    
#
# class FeedForward(tf.keras.layers.Layer):
#     def __init__(self, d_model, dff, dropout_rate=0.1):
#         super().__init__()
#         self.seq = tf.keras.Sequential([
#         tf.keras.layers.Dense(dff, activation='relu'),
#         tf.keras.layers.Dense(d_model),
#         tf.keras.layers.Dropout(dropout_rate)
#         ])
#         self.add = tf.keras.layers.Add()
#         self.layer_norm = tf.keras.layers.LayerNormalization()
#
#     def call(self, x):
#         x = self.add([x, self.seq(x)])
#         x = self.layer_norm(x) 
#         return x
#
# class EncoderLayer(tf.keras.layers.Layer):
#     def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
#         super().__init__()
#
#         self.self_attention = GlobalSelfAttention(
#             num_heads=num_heads,
#             key_dim=d_model,
#             dropout=dropout_rate)
#
#         self.ffn = FeedForward(d_model, dff)
#
#     def call(self, x):
#         x = self.self_attention(x)
#         x = self.ffn(x)
#         return x
#
# class Encoder(tf.keras.layers.Layer):
#     def __init__(self, *, num_layers, d_model, num_heads,
#                 dff, vocab_size, dropout_rate=0.1):
#         super().__init__()
#
#         self.d_model = d_model
#         self.num_layers = num_layers
#
#         self.pos_embedding = PositionalEmbedding(
#             vocab_size=vocab_size, d_model=d_model)
#
#         self.enc_layers = [
#             EncoderLayer(d_model=d_model,
#                         num_heads=num_heads,
#                         dff=dff,
#                         dropout_rate=dropout_rate)
#             for _ in range(num_layers)]
#         self.dropout = tf.keras.layers.Dropout(dropout_rate)
#
#     def call(self, x):
#         # `x` is token-IDs shape: (batch, seq_len)
#         x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.
#
#         # Add dropout.
#         x = self.dropout(x)
#
#         for i in range(self.num_layers):
#             x = self.enc_layers[i](x)
#
#         return x  # Shape `(batch_size, seq_len, d_model)`.
#
# class DecoderLayer(tf.keras.layers.Layer):
#     def __init__(self,
#                 *,
#                 d_model,
#                 num_heads,
#                 dff,
#                 dropout_rate=0.1):
#         super(DecoderLayer, self).__init__()
#
#         self.causal_self_attention = CausalSelfAttention(
#             num_heads=num_heads,
#             key_dim=d_model,
#             dropout=dropout_rate)
#
#         self.cross_attention = CrossAttention(
#             num_heads=num_heads,
#             key_dim=d_model,
#             dropout=dropout_rate)
#
#         self.ffn = FeedForward(d_model, dff)
#
#     def call(self, x, context):
#         x = self.causal_self_attention(x=x)
#         x = self.cross_attention(x=x, context=context)
#
#         # Cache the last attention scores for plotting later
#         self.last_attn_scores = self.cross_attention.last_attn_scores
#
#         x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
#         return x
#     
# class Decoder(tf.keras.layers.Layer):
#     def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
#                 dropout_rate=0.1):
#         super(Decoder, self).__init__()
#
#         self.d_model = d_model
#         self.num_layers = num_layers
#
#         self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
#                                                 d_model=d_model)
#         self.dropout = tf.keras.layers.Dropout(dropout_rate)
#         self.dec_layers = [
#             DecoderLayer(d_model=d_model, num_heads=num_heads,
#                         dff=dff, dropout_rate=dropout_rate)
#             for _ in range(num_layers)]
#
#         self.last_attn_scores = None
#
#     def call(self, x, context):
#         # `x` is token-IDs shape (batch, target_seq_len)
#         x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)
#
#         x = self.dropout(x)
#
#         for i in range(self.num_layers):
#             x  = self.dec_layers[i](x, context)
#
#         self.last_attn_scores = self.dec_layers[-1].last_attn_scores
#
#         # The shape of x is (batch_size, target_seq_len, d_model).
#         return x
#
# class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
#   def __init__(self, d_model, warmup_steps=4000):
#     super().__init__()
#
#     self.d_model = d_model
#     self.d_model = tf.cast(self.d_model, tf.float32)
#
#     self.warmup_steps = warmup_steps
#
#   def __call__(self, step):
#     step = tf.cast(step, dtype=tf.float32)
#     arg1 = tf.math.rsqrt(step)
#     arg2 = step * (self.warmup_steps ** -1.5)
#
#     return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
#
# def positional_encoding(length, depth):
#     depth = depth/2
#
#     positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
#     depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)
#
#     angle_rates = 1 / (10000**depths)         # (1, depth)
#     angle_rads = positions * angle_rates      # (pos, depth)
#
#     pos_encoding = np.concatenate(
#         [np.sin(angle_rads), np.cos(angle_rads)],
#         axis=-1) 
#
#     return tf.cast(pos_encoding, dtype=tf.float32)
#
# def masked_loss(label, pred):
#     mask = label != 0
#     loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
#         from_logits=True, reduction='none')
#     loss = loss_object(label, pred)
#
#     mask = tf.cast(mask, dtype=loss.dtype)
#     loss *= mask
#
#     loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
#     return loss
#
#
# def masked_accuracy(label, pred):
#     pred = tf.argmax(pred, axis=2)
#     label = tf.cast(label, pred.dtype)
#     match = label == pred
#
#     mask = label != 0
#
#     match = match & mask
#
#     match = tf.cast(match, dtype=tf.float32)
#     mask = tf.cast(mask, dtype=tf.float32)
#     return tf.reduce_sum(match)/tf.reduce_sum(mask)
#
# num_layers = 4
# d_model = 128
# dff = 512
# num_heads = 8
# dropout_rate = 0.1
#
# learning_rate = CustomSchedule(d_model)
#
# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
#                                      epsilon=1e-9)
#
# # transformer = Transformer(
# #     num_layers=num_layers,
# #     d_model=d_model,
# #     num_heads=num_heads,
# #     dff=dff,
# #     input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
# #     target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
# #     dropout_rate=dropout_rate)
# #
# # output = transformer((pt, en))
# #
# # print(en.shape)
# # print(pt.shape)
# # print(output.shape)
# #
# # transformer.summary()
# #
# # transformer.compile(
# #     loss=masked_loss,
# #     optimizer=optimizer,
# #     metrics=[masked_accuracy])
# #
# # transformer.fit(train_batches,
# #                 epochs=20,
# #                 validation_data=val_batches)
# #
#
# # Initialize the preprocessor
# preprocessor = ChordLyricProcessor()
#
# # Get the dataset using the preprocessor
# dataset = preprocessor.get_dataset()
#
# # Define vocabulary sizes based on your preprocessor's tokenizers
# input_vocab_size = preprocessor.lyric_tokenizer.vocabulary_size()
# target_vocab_size = 12  # Update with your target vocabulary size
#
# # Create the transformer with adapted vocabulary sizes
# transformer = Transformer(
#     num_layers=num_layers,
#     d_model=d_model,
#     num_heads=num_heads,
#     dff=dff,
#     input_vocab_size=input_vocab_size,
#     target_vocab_size=target_vocab_size,
#     dropout_rate=dropout_rate)
#
# # Define optimizer and learning rate scheduler
# learning_rate = CustomSchedule(d_model)
# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
#
# # Compile the model
# transformer.compile(
#     loss=masked_loss,
#     optimizer=optimizer,
#     metrics=[masked_accuracy])
#
# # Train the model
# transformer.fit(dataset, epochs=20)
#

