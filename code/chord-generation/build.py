import os
import numpy as np
from music21 import *
from dataloader import get_filenames, convert_files
from model import build_model
from config import *
from tqdm import trange

chord_dictionary = [
    'R', 'Cm', 'C', 'C#m', 'C#', 'Dm', 'D', 'D#m', 'D#',
    'Em', 'E', 'Fm', 'F', 'F#m', 'F#', 'Gm', 'G', 'G#m', 'G#',
    'Am', 'A', 'A#m', 'A#', 'Bm', 'B'
]

def predict(song, model):
    """Predict chord sequences from melody using the given model."""
    chord_list = []
    segments = [song[i:i+4] for i in range(0, len(song), 4)]

    for segment in segments:
        melody = np.array([np.array(part) for part in segment])[np.newaxis, ...]
        predictions = model.predict(melody)[0]
        chords = [chord_dictionary[i.argmax()] for i in predictions]
        chord_list.extend(chords)

    return chord_list[:len(song)]  # Trim to match song length if necessary

def export_music(score, chord_list, gap_list, filename):
    """Export the processed music to a .mxl file."""
    filename = os.path.splitext(os.path.basename(filename))[0]
    output_score = stream.Score()

    for i, (m, chord, gap) in enumerate(zip(score.recurse().getElementsByClass(stream.Measure), chord_list, gap_list)):
        if chord != 'R':
            chord_symbol = harmony.ChordSymbol(chord).transpose(-gap.semitones)
            m.insert(0, chord_symbol)
        output_score.append(m)

    output_score.write('mxl', fp=os.path.join(OUTPUTS_PATH, f'{filename}.mxl'))

if __name__ == '__main__':
    model = build_model(weights_path='weights.keras')
    filenames = get_filenames(extensions=INPUTS_PATH)
    data_corpus = convert_files(filenames, fromDataset=False)

    for idx in trange(len(data_corpus)):
        melody_vecs, gap_list, score, filename = data_corpus[idx]
        chord_list = predict(melody_vecs, model)
        export_music(score, chord_list, gap_list, filename)
