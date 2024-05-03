import os
import pickle
from tqdm import tqdm
import numpy as np
from music21 import *
from config import DATASET_PATH, EXTENSION, CORPUS_PATH

def safe_invert_chord(chord_symbol, inversion):
    try:
        return chord_symbol.inversion(inversion)
    except Exception as e:
        print(f"Failed to invert {chord_symbol.figure}: {e}")
        return chord_symbol  # Return the original chord if inversion fails

def get_transposition_interval(key_signature):
    """Calculate the interval needed to transpose a piece to C major/A minor."""
    if isinstance(key_signature, key.KeySignature):
        key_signature = key_signature.asKey()

    try:
        tonic = key_signature.tonic if key_signature.mode == 'major' else key_signature.parallel.tonic
    except AttributeError:
        return interval.Interval(0)
    
    return interval.Interval(tonic, pitch.Pitch('C'))

def get_filenames(directory, extensions=None):
    """Retrieve file paths from the specified directory that match the given extensions."""
    filenames = []
    for dirpath, _, files in os.walk(directory):
        for file in files:
            if extensions is None or os.path.splitext(file)[1] in extensions:
                filenames.append(os.path.join(dirpath, file))
    return filenames

def harmony_to_index(chord_element):
    """Convert a chord element to an index representing its base note and quality."""
    pitches = sorted(note.pitch.midi for note in chord_element.notes)
    base_note = pitches[0] % 12
    quality = 0 if (pitches[min(1, len(pitches) - 1)] - pitches[0]) <= 3 else 1
    return base_note * 2 + quality

def read_melody(score):
    """Extract melody vectors, chords, and transposition intervals from a musical score."""
    melody_vectors = []
    chords = []
    transpositions = []
    last_chord = 0
    last_key_signature = key.KeySignature(0)

    for measure in score.recurse().getElementsByClass(stream.Measure):
        vector = [0] * 12
        key_signature = measure.keySignature or last_key_signature
        transposition = get_transposition_interval(key_signature)

        transpositions.append(transposition)
        this_chord = None

        for element in measure.notesAndRests:
            if isinstance(element, note.Note):
                midi_pitch = element.transpose(transposition).pitch.midi
            elif isinstance(element, chord.Chord):
                notes = sorted(n.transpose(transposition).pitch.midi for n in element.notes)
                midi_pitch = notes[-1]
            elif isinstance(element, harmony.ChordSymbol):
                inverted_chord_symbol = safe_invert_chord(element, 1)  # Example: trying to invert to the first inversion
                transposed_chord = inverted_chord_symbol.transpose(transposition)
                if sum(vector) == 0:  # if no notes have been processed in this measure
                    this_chord = harmony_to_index(transposed_chord) + 1
                    last_chord = this_chord
                continue
            else:
                continue

            vector[midi_pitch % 12] += element.quarterLength

        if sum(vector):
            melody_vectors.append(np.array(vector) / sum(vector))

        chords.append(this_chord or last_chord)

    return melody_vectors, chords, transpositions

def convert_files(filenames, is_dataset=True):
    """Convert music files into a data corpus with melody vectors and chords."""
    print(f'\nConverting {len(filenames)} files...')
    failed_files = []
    data_corpus = []

    for filename in tqdm(filenames):
        try:
            score = converter.parse(filename).parts[0]
            original_score = deepcopy(score) if not is_dataset else None
            melody_vectors, chord_text, transpositions = read_melody(score)
            data_corpus.append((melody_vectors, chord_text, transpositions, original_score, filename) if not is_dataset else (melody_vectors, chord_text))
        except Exception as e:
            failed_files.append((filename, str(e)))

    success_count = len(filenames) - len(failed_files)
    print(f'Successfully converted {success_count} files.')
    if failed_files:
        print(f'Failed to process {len(failed_files)} files:')
        for failed_file in failed_files:
            print(failed_file)

    if is_dataset:
        with open(CORPUS_PATH, 'wb') as f:
            pickle.dump(data_corpus, f)
    else:
        return data_corpus

if __name__ == '__main__':
    filenames = get_filenames(DATASET_PATH, EXTENSION)
    convert_files(filenames)
