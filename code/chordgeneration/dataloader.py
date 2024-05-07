import os
import pickle
import numpy as np
from copy import deepcopy
from tqdm import trange
from music21 import *
from config import *

def transpose_melody(melody, transpose_interval):
    """
    Transpose a melody by a given interval.

    Args:
        melody (list): List of 12-dimensional melody vectors.
        transpose_interval (int): Interval by which to transpose.

    Returns:
        list: Transposed melody.
    """
    transposed_melody = []
    for vec in melody:
        transposed_vec = np.roll(vec, transpose_interval)
        transposed_melody.append(transposed_vec)
    return transposed_melody

def add_noise(melody, noise_factor=0.1):
    """
    Add Gaussian noise to a melody.

    Args:
        melody (list): List of 12-dimensional melody vectors.
        noise_factor (float): Standard deviation of Gaussian noise.

    Returns:
        list: Melody with added noise.
    """
    noisy_melody = []
    for vec in melody:
        noisy_vec = vec + noise_factor * np.random.randn(len(vec))
        noisy_vec = np.clip(noisy_vec, 0, 1)
        noisy_vec = noisy_vec / np.sum(noisy_vec)  # Re-normalize
        noisy_melody.append(noisy_vec)
    return noisy_melody

def ks2gap(key_signature):
    """
    Convert a KeySignature or Key object to an Interval representing the gap to C Major.

    Args:
        key_signature (music21.key.KeySignature or music21.key.Key): Input key signature.

    Returns:
        music21.interval.Interval: Interval to C Major.
    """
    if isinstance(key_signature, key.KeySignature):
        key_signature = key_signature.asKey()
    try:
        tonic = key_signature.tonic if key_signature.mode == 'major' else key_signature.parallel.tonic
    except Exception:
        return interval.Interval(0)

    return interval.Interval(tonic, pitch.Pitch('C'))

def get_filenames(input_dir):
    """
    Get all filenames in a directory, optionally filtering by extension.

    Args:
        input_dir (str): Path to the directory.

    Returns:
        list: List of file paths.
    """
    filenames = []
    for dirpath, _, filelist in os.walk(input_dir):
        for this_file in filelist:
            if input_dir == DATASET_PATH and os.path.splitext(this_file)[-1] not in EXTENSION:
                continue
            filenames.append(os.path.join(dirpath, this_file))
    return filenames

def harmony2idx(harmonic_element):
    """
    Convert a harmony chord to an index.

    Args:
        harmonic_element (music21.harmony.ChordSymbol): Harmony chord element.

    Returns:
        int: Chord index.
    """
    pitch_list = sorted([sub_ele.pitch.midi for sub_ele in harmonic_element.notes])
    bass_note = pitch_list[0] % 12
    quality = pitch_list[min(1, len(pitch_list) - 1)] - pitch_list[0]

    quality = 0 if quality <= 3 else 1
    return bass_note * 2 + quality

def melody_reader(score):
    """
    Extract melody vectors and chord data from a score.

    Args:
        score (music21.stream.Score): Input score.

    Returns:
        tuple: (melody vectors, chord indices, gap list)
    """
    melody_vecs = []
    chord_list = []
    gap_list = []
    last_chord = 0
    last_key_signature = key.KeySignature(0)

    for measure in score.recurse(classFilter=stream.Measure):
        vec = [0] * 12
        gap = ks2gap(measure.keySignature or last_key_signature)
        gap_list.append(gap)
        this_chord = None

        for element in measure:
            if isinstance(element, note.Note):
                token = element.transpose(gap).pitch.midi
            elif isinstance(element, chord.Chord) and not isinstance(element, harmony.ChordSymbol):
                notes = sorted([n.transpose(gap).pitch.midi for n in element.notes])
                token = notes[-1]
            elif isinstance(element, harmony.ChordSymbol) and sum(vec) == 0:
                this_chord = harmony2idx(element.transpose(gap)) + 1
                last_chord = this_chord
                continue
            else:
                continue

            vec[token % 12] += float(element.quarterLength)

        if sum(vec) != 0:
            vec = np.array(vec) / sum(vec)

        melody_vecs.append(vec)
        chord_list.append(this_chord if this_chord is not None else last_chord)

    return melody_vecs, chord_list, gap_list

def convert_files(filenames, from_dataset=True, augment=False, num_augmentations=2):
    """
    Convert a list of music files into melody vectors and chord data.

    Args:
        filenames (list): List of file paths.
        from_dataset (bool): Whether the files are from a dataset.
        augment (bool): Whether to apply data augmentation.
        num_augmentations (int): Number of augmented versions to generate per file.

    Returns:
        list: Converted data corpus.
    """
    print(f'\nConverting {len(filenames)} files...')
    failed_list = []
    data_corpus = []

    for filename_idx in trange(len(filenames)):
        filename = filenames[filename_idx]
        try:
            score = converter.parse(filename).parts[0]
            original_score = deepcopy(score) if not from_dataset else None

            melody_vecs, chord_txt, gap_list = melody_reader(score)

            if from_dataset:
                data_corpus.append([(melody_vecs, chord_txt)])
                if augment:
                    for _ in range(num_augmentations):
                        interval = np.random.choice(range(-3, 4))  # Transpose between -3 and +3 semitones
                        transposed_melody = transpose_melody(melody_vecs, interval)
                        noisy_melody = add_noise(transposed_melody, noise_factor=0.05)
                        data_corpus.append([(noisy_melody, chord_txt)])
            else:
                data_corpus.append((melody_vecs, gap_list, original_score, filename))
                if augment:
                    for _ in range(num_augmentations):
                        interval = np.random.choice(range(-3, 4))
                        transposed_melody = transpose_melody(melody_vecs, interval)
                        noisy_melody = add_noise(transposed_melody, noise_factor=0.05)
                        data_corpus.append((noisy_melody, gap_list, original_score, filename))
        except Exception as e:
            failed_list.append((filename, e))

    print(f'Successfully converted {len(filenames) - len(failed_list)} files.')
    if failed_list:
        print(f'Failed numbers: {len(failed_list)}')
        print('Failed to process:')
        for failed_file in failed_list:
            print(failed_file)

    if from_dataset:
        with open(CORPUS_PATH, "wb") as filepath:
            pickle.dump(data_corpus, filepath)
    else:
        return data_corpus


if __name__ == '__main__':
    filenames = get_filenames(DATASET_PATH)
    convert_files(filenames, augment=False, num_augmentations=2)
