import os
import warnings
from music21 import *
from tqdm import trange
from model import MusicModelTrainer
from dataloader import get_filenames, convert_files
from config import *
import numpy as np

CHORD_DICTIONARY = ['R', 'Cm', 'C', 'C#m', 'C#', 'Dm', 'D', 'D#m', 'D#', 'Em', 'E', 'Fm', 'F',
                    'F#m', 'F#', 'Gm', 'G', 'G#m', 'G#', 'Am', 'A', 'A#m', 'A#', 'Bm', 'B']

class ChordPredictor:
    def __init__(self, model_weights='weights.keras'):
        """
        Initialize the chord predictor.

        Args:
            model_weights (str): Path to the model weights.
        """
        self.model = MusicModelTrainer().build_model()

    def predict_chords(self, song):
        """
        Predict chords for a given melody.

        Args:
            song (list): List of melody vectors.

        Returns:
            list: List of predicted chords.
        """
        num_segments = len(song) // 4
        chord_list = []

        for idx in range(num_segments):
            melody = np.array(song[idx * 4: (idx + 1) * 4])[np.newaxis, ...]
            net_output = self.model.predict(melody, verbose=0)[0]
            chord_list.extend([CHORD_DICTIONARY[chord_idx] for chord_idx in net_output.argmax(axis=1)])

        remaining = len(song) % 4
        if remaining > 0:
            melody = np.array(song[-4:])[np.newaxis, ...]
            net_output = self.model.predict(melody, verbose=0)[0]
            chord_list.extend([CHORD_DICTIONARY[net_output[idx].argmax()] for idx in range(-remaining, 0)])

        return chord_list

def ensure_well_formed(score):
    """
    Ensure that a score is well-formed by fixing common issues.
    Args:
        score (music21.stream.Score): Input score stream.

    Returns:
        music21.stream.Score: Well-formed score.
    """
    issues_found = False
    for part in score.parts:
        for measure in part.getElementsByClass(stream.Measure):
            if not measure.timeSignature:
                measure.insert(0, meter.TimeSignature('4/4')) 
                issues_found = True
            if not measure.keySignature:
                measure.insert(0, key.KeySignature(0))
                issues_found = True
            if not measure.notes:
                measure.append(note.Rest())
                issues_found = True
        if not part.flat.timeSignature:
            part.insert(0, meter.TimeSignature('4/4'))
            issues_found = True
        if not part.flat.keySignature:
            part.insert(0, key.KeySignature(0))
            issues_found = True

    score.makeMeasures(inPlace=True)

    # if issues_found:
    #     warnings.warn(f"{score} had issues and was modified; see isWellFormedNotation()")

    return score

def export_to_musicxml(score, chord_list, gap_list, filename):
    """
    Export the melody with predicted chords into a MusicXML file.

    Args:
        score (music21.stream.Score): Original music score.
        chord_list (list): List of predicted chords.
        gap_list (list): List of Interval objects representing gaps.
        filename (str): Output filename.
    """
    base_name = os.path.basename(filename).rsplit('.', 1)[0]
    harmony_list = [harmony.ChordSymbol(chord).transpose(-gap.semitones) if chord != 'R' else note.Rest()
                    for chord, gap in zip(chord_list, gap_list)]

    new_measures = []
    m_idx = 0
    for measure in score.recurse(classFilter=stream.Measure):
        new_measure_elements = [harmony_list[m_idx]] if not isinstance(harmony_list[m_idx], note.Rest) else []
        new_measure_elements.extend([n for n in measure if not isinstance(n, harmony.ChordSymbol)])
        new_measure = stream.Measure(new_measure_elements)
        new_measure.offset = measure.offset
        new_measures.append(new_measure)
        m_idx += 1

    new_measures[-1].rightBarline = bar.Barline('final')
    new_score = stream.Score(new_measures)

    # Ensure the new score is well-formed
    new_score = ensure_well_formed(new_score)
    # if not new_score.isWellFormedNotation():
    #     warnings.warn(f"{new_score} is not well-formed after adjustments; see isWellFormedNotation()", stacklevel=2)

    output_path = os.path.join(OUTPUTS_PATH, f"{base_name}.mxl")
    new_score.write('mxl', fp=output_path)

if __name__ == '__main__':
    predictor = ChordPredictor(model_weights=WEIGHTS_PATH)
    filenames = get_filenames(INPUTS_PATH)
    data_corpus = convert_files(filenames, from_dataset=False)

    for idx in trange(len(data_corpus)):
        melody_vecs, gap_list, score, filename = data_corpus[idx]
        chord_list = predictor.predict_chords(melody_vecs)
        export_to_musicxml(score, chord_list, gap_list, filename)
