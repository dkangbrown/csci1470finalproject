from utils import re

def preprocess_song(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Regular expressions to identify chords and tabs
    chord_regex = re.compile(r"[A-G][#b]?m?(maj7|maj|min7|min|7|sus4)?")
    section_regex = re.compile(r"^\[.*\]$")
    accidental_regex = re.compile(r"#|b")

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
        'm': 1,
        '+': 2,
        'dim': 3,
        '7': 4,
        'maj7': 5,
        'min7': 6,
        'sus2': 7,
        'sus4': 8
    }

    def get_root_pitch(chord):
        root_key = chord.split()[0:1]
        if accidental_regex.match(root_key[1]):
            root_key = root_key[0]+root_key[1]
        else:
            root_key = root_key[0]

        return key_to_pitch[root_key]

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

    def chord_to_vector(chord):
        chord_rel_pitch = get_rel_pitch(get_root_pitch(chord))
        
        if len(chord.split('/')) == 2:
            base_rel_pitch = get_rel_pitch(get_root_pitch(chord.split('/')[1]))
        else:
            base_rel_pitch = chord_rel_pitch

        if accidental_regex.match(chord.split()[1]):
            qual = chord.split()[2:]
        else:
            qual = chord.split()[1:]
        
        if qual.contains('add'):
            qual = qual[0:qual.index('add')]

        return [chord_rel_pitch,base_rel_pitch,qual_to_num[qual]]

    def get_chord_list(line):
        line = line.strip()
        chord_list = line.split(" ")
        chord_list = [chord_to_vector(x) for x in chord_list if x != '']
        return chord_list

    for line in lines:
        line = line.strip()

        if section_regex.match(line):
            if isbreak == 0:
                isbreak == 1
                versenum += 1
                verses[versenum] = {'text': "", 'chords': []}
        else:
            if isbreak == 1:
                isbreak == 0
                if chord_regex.match(line):
                    verses[versenum]['chords'] += get_chord_list(line)
            else:
                if chord_regex.match(line):
                    verses[versenum]['chords'] += get_chord_list(line)
                else:
                    verses[versenum]['text'] += line + " "

        # elif tab_regex.match(line):
        #     # Collect tab lines under the current section
        #     if 'tabs' not in song_data:
        #         song_data['tabs'] = {}
        #     if current_section not in song_data['tabs']:
        #         song_data['tabs'][current_section] = []
        #     song_data['tabs'][current_section].append(line)
        # else:
        #     if chord_regex.search(line):
        #         chords = line
        #     else:
        #         lyrics = line

        #     if chords and lyrics:
        #         song_data['lyrics_and_chords'][current_section].append({'chords': chords, 'lyrics': lyrics})
        #         chords = ""  # Reset after pairing
        #         lyrics = ""

    return song_data

if __name__ == '__main__':
    file_path = 'song.txt'
    preprocessed_data = preprocess_song(file_path)
    print(preprocessed_data)
