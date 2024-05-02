import re
import os
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import numpy as np


def preprocess_song(file_path):
    # Regular expressions to identify chords, sections, and accidental notes
    chord_regex = re.compile(r"^[A-G][#b]?[0-9]?[a-zA-Z0-9#]*$")
    section_regex = re.compile(r"^\[.*\]$")
    accidental_regex = re.compile(r"^\w[#b]")

    # Mapping pitch names to numbers
    key_to_pitch = {
        'A': 0, 'A#': 1, 'Bb': 1, 'B': 2, 'Cb': 2, 'B#': 3, 'C': 3, 'C#': 4, 'Db': 4,
        'D': 5, 'D#': 6, 'Eb': 6, 'E': 7, 'Fb': 7, 'E#': 8, 'F': 8, 'F#': 9, 'Gb': 9,
        'G': 10, 'G#': 11, 'Ab': 11
    }

    # Chord quality to number mapping
    qual_to_num = {
        '': 0, '5': 0, 'M': 0, 'maj7': 1, 'maj': 1, 'M7': 1, '7': 2, 'm': 3, 'mmaj7': 3,
        'min7': 4, 'm7': 4, 'dim': 5, 'dim7': 5, '+': 6, '+5': 6, 'aug': 6,
        'sus2': 7, '7sus2': 7, '2': 7, 'sus4': 8, '7sus4': 8, '4': 8
    }

    def get_root_pitch(chord):
        root_note = chord[0] + chord[1] if accidental_regex.match(chord) else chord[0]
        return key_to_pitch[root_note]

    def get_rel_pitch(pitch, key_pitch):
        return (pitch - key_pitch) % 12

    def chord_to_vector(chord, key_pitch):
        base, *qual = chord.split('/')
        root_pitch = get_rel_pitch(get_root_pitch(base), key_pitch)
        qual = qual[0] if qual else ''
        base_pitch = root_pitch if not qual else get_rel_pitch(get_root_pitch(qual), key_pitch)
        qual_number = qual_to_num[re.sub(r'\d+', '', qual.split('add')[0])]
        return [root_pitch, base_pitch, qual_number]

    verses = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

    key = lines[0].strip().split(' ')[1]
    key_pitch = get_root_pitch(key)
    versenum = -1

    # Initialize the first verse if no sections are provided
    verses.append({'text': "", 'chords': []})
    versenum = 0

    for line in lines[1:]:
        clean_line = re.sub(r"[|\(\)\-%\t\\/*@,x\d]", " ", line).strip()
        if section_regex.match(clean_line):
            verses.append({'text': "", 'chords': []})
            versenum = len(verses) - 1
        else:
            is_chord_line = all(chord_regex.match(word) for word in clean_line.split())
            if is_chord_line and versenum >= 0:
                verses[versenum]['chords'] += [chord_to_vector(word, key_pitch) for word in clean_line.split()]
            else:
                verses[versenum]['text'] += clean_line + " "

    return [verse for verse in verses if verse['chords'] or verse['text']]


def preprocess_directory(directory_path):
    preprocessed_pairs = []
    file_name_regex = re.compile(r"^([A-R]|[a-r])")
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".txt") and file_name_regex.match(file_name):
            file_path = os.path.join(directory_path, file_name)
            song_data = preprocess_song(file_path)
            preprocessed_pairs.extend(song_data)
    return preprocessed_pairs


song_files_directory = "../data/chord-lyric-text/"
preprocessed_pairs = preprocess_directory(song_files_directory)
print("Number of songs:", len(preprocessed_pairs))


def transpose_chords(chords, semitone_shift):
    new_chords = []
    for root, base, quality in chords:
        transposed_root = (root + semitone_shift) % 12
        transposed_base = (base + semitone_shift) % 12
        new_chords.append([transposed_root, transposed_base, quality])
    return new_chords


# Example of applying the transposition
augmented_data = []
for song in preprocessed_pairs:
    original_chords = song['chords']
    for shift in range(1, 12):  # Generate 11 new songs per original by shifting chords
        new_chords = transpose_chords(original_chords, shift)
        augmented_song = {'text': song['text'], 'chords': new_chords}
        augmented_data.append(augmented_song)

# Now your dataset includes the original and the augmented versions
full_dataset = preprocessed_pairs + augmented_data
print("Number of songs after augmentation:", len(full_dataset))


print(preprocessed_pairs[0])

# Define maximum sequence lengths for padding
MAX_SEQUENCE_LENGTH_LYRICS = 250
MAX_SEQUENCE_LENGTH_CHORDS = 250

# Initialize the lyric tokenizer
lyric_tokenizer = TextVectorization(output_mode='int', output_sequence_length=MAX_SEQUENCE_LENGTH_LYRICS)
lyric_tokenizer.adapt([item['text'] for item in preprocessed_pairs if item['text']])  # Adapt tokenizer only on non-empty lyrics
print("Vocabulary size:", len(lyric_tokenizer.get_vocabulary()))

def tokenize_lyrics(text):
    return lyric_tokenizer(text) if text else np.zeros((MAX_SEQUENCE_LENGTH_LYRICS,), dtype=np.int32)

def pad_chords(chords, max_len=MAX_SEQUENCE_LENGTH_CHORDS):
    if chords:
        padded_chords = tf.keras.preprocessing.sequence.pad_sequences([chords], maxlen=max_len, padding='post', dtype='int32', value=[0, 0, 0])
        return padded_chords[0]
    else:
        return np.zeros((max_len, 3), dtype=np.int32)

def prepare_dataset(preprocessed_data, batch_size=32):
    def generator():
        for item in preprocessed_data:
            if item['text'] and item['chords']:
                tokenized_lyrics = tokenize_lyrics(item['text'])
                padded_chords = pad_chords(item['chords'])
                yield (tokenized_lyrics.numpy(), padded_chords)
            else:
                yield (np.zeros((MAX_SEQUENCE_LENGTH_LYRICS,), dtype=np.int32), 
                       np.zeros((MAX_SEQUENCE_LENGTH_CHORDS, 3), dtype=np.int32))

    dataset = tf.data.Dataset.from_generator(
        generator, 
        output_types=(tf.int32, tf.int32), 
        output_shapes=((MAX_SEQUENCE_LENGTH_LYRICS,), (MAX_SEQUENCE_LENGTH_CHORDS, 3))
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Use the dataset
dataset = prepare_dataset(preprocessed_pairs)

for lyrics, chords in dataset.take(1):
    print("Lyrics batch shape:", lyrics.shape)
    print("Chords batch shape:", chords.shape)
    print("Sample lyrics:", lyrics[4])
    print("Sample chords:", chords[6])


def get_dataset():
    preprocessed_pairs = preprocess_directory(song_files_directory)
    return prepare_dataset(preprocessed_pairs)
