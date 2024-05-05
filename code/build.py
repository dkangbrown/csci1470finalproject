import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout

import os
from utils import re
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import string

def preprocess_song(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Regular expressions to identify chords and tabs
    chord_regex = re.compile(r"^[A-G][#b]?(7|5|M|maj7|maj|M7|mmaj7|min7|m7|m|min|dim|dim7|aug|\+|sus2|sus4|7sus2|7sus4)?(add)?[0-9]*/?[A-G]?[#b]?$")
    section_regex = re.compile(r"^(\(|\[)?[\#]?(chorus|Chorus|CHORUS|verse|Verse|VERSE|intro|Intro|INTRO|outro|Outro|OUTRO|bridge|Bridge|BRIDGE|interlude|Interlude|INTERLUDE|instrumental|Instrumental|INSTRUMENTAL|solo|Solo|SOLO)*( )?[0-9]*(\:|\.)?(\]|\))?$")
    accidental_regex = re.compile(r"^\w[#b]")

    current_section = None
    key_to_pitch = {
        'A': 0,
        'A#': 1,
        'Bb': 1,
        'B': 2,
        'Cb': 2,
        'B#': 3,
        'C': 3,
        'C#': 4,
        'Db': 4,
        'D': 5,
        'D#': 6,
        'Eb': 6,
        'E': 7,
        'Fb': 7,
        'E#': 8,
        'F': 8,
        'F#': 9,
        'Gb': 9,
        'G': 10,
        'G#': 11,
        'Ab': 11
    }
    qual_to_num = {
        '': 0,
        '5': 0,
        'M' : 0,
        'maj7': 1,
        'maj': 1,
        'M7': 1,
        '7': 2,
        'm': 3,
        'mmaj7': 3,
        'min7': 4,
        'm7': 4,
        'dim': 5,
        'dim7':5,
        '+': 6,
        '+5': 6,
        'aug': 6,
        'sus2': 7,
        '7sus2': 7,
        '2': 7,
        'sus4': 8,
        '7sus4': 8,
        '4': 8
    }

    def get_root_pitch(chord):
        if accidental_regex.match(chord):
            root_note = chord[0]+chord[1]
        else:
            root_note = chord[0]
        
        # print(f"root_note: {root_note}")

        return key_to_pitch[root_note]

    verses = []
    key = lines[0].strip().split(' ')[1]
    # print(f"key: {key}")
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

        # print(f"root: {chord_rel_pitch}")
        # print(f"base: {base_rel_pitch}")
        # print(f"qual: {qual}")

        return [chord_rel_pitch, base_rel_pitch, qual_to_num[qual]]

    def get_chord_list(line):
        line = line.strip()
        chord_list = line.split(" ")
        chord_list = [x for x in chord_list if x]
        # print(f"chord list: {chord_list}")
        chord_list = [chord_to_vector(x) for x in chord_list]
        return chord_list

    for line in lines:
        line = line.strip().replace('|',' ').replace('(',' ').replace(')',' ').replace('-',' ').replace('%',' ').replace('\t',' ').replace('\\','/').replace('/ ',' ').replace('*',' ').replace('@',' ').replace(',  ','   ').replace('x2',' ').replace('x3',' ').replace('x4',' ').replace('x5',' ').replace('x6',' ').replace('x7',' ').replace('x8',' ')
        # print(f"line: {line}")
        if section_regex.match(line):
            # print(f"section regex match; isbreak = {isbreak}")
            if isbreak == 0:
                isbreak = 1
                versenum += 1
                # print(f"versenum: {versenum}")
                verses.append({'text': "", 'chords': []})
        else:
            # print(f"No section regex match; isbreak = {isbreak}")
            if isbreak == 1:
                isbreak = 0
            is_chord_line = True
            for word in line.split():
                if not(is_chord_line and chord_regex.match(word)):
                    is_chord_line = False
            if is_chord_line:
                verses[versenum]['chords'] += get_chord_list(line)
                # print(verses[versenum]['chords'])
            else:
                verses[versenum]['text'] += line + " "

    # FORMAT OF THE OUTPUT:
    # list (for each song) of dictionaries (for each verse) with keys 'text' and 'chords'
    # chords are in the format [0-11 (root pitch), 0-11 (base pitch), 0-7 (quality)]
    return [x for x in verses if x['chords']]

Path = "../data/chord-lyric-text/"
filelist = os.listdir(Path)
preprocessed_pairs = []
file_name = re.compile(r"^([A-R]|[a-r])")
for i in filelist:
    if i.endswith(".txt"):
        # print(i)
        song_data = preprocess_song(Path + i)
        # print(song_data)
        preprocessed_pairs += song_data
print(preprocessed_pairs[0:5])

# def prepare_data(texts, chord_vectors):
#     """ Prepare data for model input from texts and chord vectors. """
#     input_ids, attention_masks, targets = [], [], []

#     for text, chords in zip(texts, chord_vectors):
#         inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=512)
#         input_ids.append(inputs['input_ids'])
#         attention_masks.append(inputs['attention_mask'])
#         encoded_chords = encode_chords(chords)
#         targets.append(tf.convert_to_tensor(encoded_chords, dtype=tf.int32))

#     return tf.concat(input_ids, axis=0), tf.concat(attention_masks, axis=0), tf.concat(targets, axis=0)

def encode_chord(root_pitch, base_pitch, quality):
    """ Encode chord components into a single integer. """
    return root_pitch * 96 + base_pitch * 8 + quality

def decode_chord(encoded_chord):
    """ Decode an encoded chord integer back to its components. """
    root_pitch = encoded_chord // 96
    base_pitch = (encoded_chord % 96) // 8
    quality = encoded_chord % 8
    return (root_pitch, base_pitch, quality)

# Example to demonstrate encoding and decoding
root_pitch = 5  # D
base_pitch = 7  # F
quality = 2     # '7' in your qual_to_num dictionary
encoded_chord = encode_chord(root_pitch, base_pitch, quality)
decoded_chord = decode_chord(encoded_chord)

print("Encoded Chord:", encoded_chord)
print("Decoded Chord:", decoded_chord)

def encode_chords(chords):
    """ Convert a list of chord vectors to encoded values. """
    return [str(encode_chord(chord[0], chord[1], chord[2])) for chord in chords]

def prepare_data(preprocessed_pairs):
    """ Prepare data for model input from texts and chord vectors. """
    inputs, targets = [], []

    for pair in preprocessed_pairs:
        # print(pair)
        split_input = pair.get("text").translate(str.maketrans('', '', string.punctuation)).lower().split()
        split_input.append("<STOP>")
        inputs.append(split_input)
        split_targets = encode_chords(pair.get("chords"))
        split_targets.append("<STOP>")
        targets.append(split_targets)
    
    # print(inputs, targets)
    unique_input_words = sorted(set([i for j in inputs for i in j]))
    input_vocab = {w:i for i, w in enumerate(unique_input_words)}
    # print(input_vocab)

    input_data = [list(map(lambda x: input_vocab.get(x), i)) for i in inputs]

    input_vocab_size = len(unique_input_words)

    unique_target_words = sorted(set([i for j in targets for i in j]))
    target_vocab = {w:i for i, w in enumerate(unique_target_words)}
    # print(target_vocab)

    target_data = [list(map(lambda x: target_vocab.get(x), i)) for i in targets]

    target_vocab_size = len(unique_target_words)

    return input_data, target_data, input_vocab_size, target_vocab_size

print(prepare_data(preprocessed_pairs[0:5]))

input_data, target_data, input_vocab_size, target_vocab_size = prepare_data(preprocessed_pairs)

target_inputs = [i[:-1] for i in target_data]
target_labels = [i[1:] for i in target_data]

### MODEL ###
embed_size = 64
hidden_size = 72

# Encoder
inputs = tf.keras.Input(shape=(None,))
input_embedding = tf.keras.layers.Embedding(input_vocab_size, embed_size)(inputs)
encoder_output, state = tf.keras.layers.GRU(hidden_size, return_state=True)(input_embedding)

# Decoder
targets = tf.keras.Input(shape=(None,))
target_embedding = tf.keras.layers.Embedding(target_vocab_size, embed_size)(targets)
decoder_output = tf.keras.layers.GRU(hidden_size, return_sequences=True)(target_embedding, initial_state = encoder_output)
classifier_output = tf.keras.layers.Dense(target_vocab_size, activation='softmax')(decoder_output)

model = tf.keras.Model(inputs=[inputs, targets], outputs=classifier_output)

def perplexity(logits, labels):
    return tf.exp(tf.reduce_mean(tf.keras.metrics.sparse_categorical_crossentropy(logits, labels), axis=-1))

optimizer=tf.keras.optimizers.Adam()
loss=tf.keras.losses.SparseCategoricalCrossentropy()
metrics=[perplexity]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.summary()
model.fit(x=[input_data, target_inputs], y=[target_labels], epochs=5, batch_size=32, validation_split=0.2)




# # Assuming texts and chord_vectors are already defined
# texts = ["your lyrics here"]
# chord_vectors = [[[5, 7, 2]]]  # List of lists of chord vectors
# input_ids, attention_masks, targets = prepare_data(texts, chord_vectors)
# print(input_ids)
# print(targets)

# # # Load tokenizer and base model
# # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# # bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# # # Encoder: leveraging the pre-trained BERT
# # input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
# # attention_mask = Input(shape=(None,), dtype=tf.int32, name="attention_mask")

# # encoder_outputs = bert_model(input_ids, attention_mask=attention_mask)[0]

# # # Decoder: custom dense layers to predict chords
# # x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(768, return_sequences=True))(encoder_outputs)
# # x = Dropout(0.4)(x)
# # x = Dense(768, activation='relu')(x)
# # outputs = Dense(12 * 12 * 8, activation='softmax')(x)  # 12 root pitches, 12 base pitches, 8 qualities

# # # Combine into a single model
# # model = Model(inputs=[input_ids, attention_mask], outputs=[outputs])
# # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # # Model summary
# # model.summary()


# # # Fit model
# # model.fit([input_ids, attention_masks], targets, epochs=5, batch_size=32)