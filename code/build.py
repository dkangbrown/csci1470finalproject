import re
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


class ChordLyricProcessor:
    def __init__(self, song_files_directory="../data/chord-lyric-text/", max_sequence_length_lyrics=250, max_sequence_length_chords=250):
        self.song_files_directory = song_files_directory

        self.MAX_SEQUENCE_LENGTH_LYRICS = max_sequence_length_lyrics
        self.MAX_SEQUENCE_LENGTH_CHORDS = max_sequence_length_chords

        self.key_to_pitch = {
            'A': 0, 'A#': 1, 'Bb': 1, 'B': 2, 'Cb': 2, 'B#': 3, 'C': 3, 'C#': 4, 'Db': 4,
            'D': 5, 'D#': 6, 'Eb': 6, 'E': 7, 'Fb': 7, 'E#': 8, 'F': 8, 'F#': 9, 'Gb': 9,
            'G': 10, 'G#': 11, 'Ab': 11
        }
        self.qual_to_num = {
            '': 0, '5': 0, 'M': 0, 'maj7': 1, 'maj': 1, 'M7': 1, '7': 2, 'm': 3, 'mmaj7': 3,
            'min7': 4, 'm7': 4, 'dim': 5, 'dim7': 5, '+': 6, '+5': 6, 'aug': 6,
            'sus2': 7, '7sus2': 7, '2': 7, 'sus4': 8, '7sus4': 8, '4': 8
        }

        self.accidental_regex = re.compile(r"^\w[#b]$")
        self.section_regex = re.compile(r"^\[.*\]$")
        self.chord_regex = re.compile(r"^[A-G][#b]?[0-9]?[a-zA-Z0-9#]*$")

        self.lyric_tokenizer = TextVectorization(output_mode='int', output_sequence_length=self.MAX_SEQUENCE_LENGTH_LYRICS)

        self.preprocessed_pairs = []

    def preprocess_song(self, file_path):
        def get_root_pitch(chord):
            root_note = chord[0] + chord[1] if self.accidental_regex.match(chord) else chord[0]
            return self.key_to_pitch[root_note]

        def get_rel_pitch(pitch, key_pitch):
            return (pitch - key_pitch) % 12

        def chord_to_vector(chord, key_pitch):
            base, *qual = chord.split('/')
            root_pitch = get_rel_pitch(get_root_pitch(base), key_pitch)
            qual = qual[0] if qual else ''
            base_pitch = root_pitch if not qual else get_rel_pitch(get_root_pitch(qual), key_pitch)
            qual_number = self.qual_to_num[re.sub(r'\d+', '', qual.split('add')[0])]
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
            if self.section_regex.match(clean_line):
                verses.append({'text': "", 'chords': []})
                versenum = len(verses) - 1
            else:
                is_chord_line = all(self.chord_regex.match(word) for word in clean_line.split())
                if is_chord_line and versenum >= 0:
                    verses[versenum]['chords'] += [chord_to_vector(word, key_pitch) for word in clean_line.split()]
                else:
                    verses[versenum]['text'] += clean_line + " "

        return [verse for verse in verses if verse['chords'] or verse['text']]

    def preprocess_directory(self, directory_path):
        file_name_regex = re.compile(r"^([A-R]|[a-r])")
        for file_name in os.listdir(directory_path):
            if file_name.endswith(".txt") and file_name_regex.match(file_name):
                file_path = os.path.join(directory_path, file_name)
                song_data = self.preprocess_song(file_path)
                self.preprocessed_pairs.extend(song_data)

    def tokenize_lyrics(self, text):
        return self.lyric_tokenizer(text) if text else np.zeros((self.MAX_SEQUENCE_LENGTH_LYRICS,), dtype=np.int32)

    def pad_chords(self, chords, max_len=None):
        max_len = max_len or self.MAX_SEQUENCE_LENGTH_CHORDS
        if chords:
            padded_chords = tf.keras.preprocessing.sequence.pad_sequences([chords], maxlen=max_len, padding='post', dtype='int32', value=[0, 0, 0])
            return padded_chords[0]
        else:
            return np.zeros((max_len, 3), dtype=np.int32)

    def prepare_dataset(self, batch_size=32):
        def generator():
            for item in self.preprocessed_pairs:
                if item['text'] and item['chords']:
                    tokenized_lyrics = self.tokenize_lyrics(item['text'])
                    padded_chords = self.pad_chords(item['chords'])
                    yield (tokenized_lyrics.numpy(), padded_chords)
                else:
                    yield (np.zeros((self.MAX_SEQUENCE_LENGTH_LYRICS,), dtype=np.int32),
                           np.zeros((self.MAX_SEQUENCE_LENGTH_CHORDS, 3), dtype=np.int32))

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=(tf.int32, tf.int32),
            output_shapes=((self.MAX_SEQUENCE_LENGTH_LYRICS,), (self.MAX_SEQUENCE_LENGTH_CHORDS, 3))
        )
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def get_dataset(self):
        self.preprocess_directory(self.song_files_directory)
        self.lyric_tokenizer.adapt([item['text'] for item in self.preprocessed_pairs if item['text']])
        return self.prepare_dataset()


processor = ChordLyricProcessor()
dataset = processor.get_dataset()

for lyrics, chords in dataset.take(1):
    print("Lyrics batch shape:", lyrics.shape)
    print("Chords batch shape:", chords.shape)
    print("Sample lyrics:", lyrics[4])
    print("Sample chords:", chords[6])
