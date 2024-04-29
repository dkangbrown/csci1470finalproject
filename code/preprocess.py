from utils import re

def preprocess_song(file_path):
    print("preprocess starting")
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Regular expressions to identify chords and tabs
    chord_regex = re.compile(r"\s*[A-G][#b]?m?(maj7|maj|min7|min|7|sus2|sus4)?")
    section_regex = re.compile(r"^(\[|#)(chorus|Chorus|CHORUS|verse|Verse|VERSE|intro|Intro|INTRO|outro|Outro|OUTRO|bridge|Bridge|BRIDGE|interlude|Interlude|INTERLUDE|instrumental|Instrumental|INSTRUMENTAL|solo|Solo|SOLO)*( )?[0-9]*(\]|\:|\.)?$")
    accidental_regex = re.compile(r"[#b]")

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
        'm7': 6,
        'sus2': 7,
        'sus4': 8
    }

    def get_root_pitch(chord):
        root_note = chord[0:2]
        print(f"root_note: {root_note}")
        if len(root_note) > 1:
            if accidental_regex.match(root_note[1]):
                root_note = root_note[0]+root_note[1]
            else:
                root_note = root_note[0]
        else:
            root_note = root_note[0]
        
        print(f"root_note: {root_note}")

        return key_to_pitch[root_note]

    verses = []
    key = lines[0].strip().split(' ')[1]
    print(f"key: {key}")
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
        
        if len(chord.split('/')) == 2:
            base_rel_pitch = get_rel_pitch(get_root_pitch(chord.split('/')[1]))
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

        print(f"pitch: {chord_rel_pitch}")
        print(f"rel: {base_rel_pitch}")
        print(f"qual: {qual}")

        return [chord_rel_pitch, base_rel_pitch, qual_to_num[qual]]

    def get_chord_list(line):
        print(f"getting chord list of {line}")
        line = line.strip()
        chord_list = line.split(" ")
        chord_list = [x for x in chord_list if x]
        print(f"chord list: {chord_list}")
        chord_list = [chord_to_vector(x) for x in chord_list]
        return chord_list

    for line in lines:
        line = line.strip()
        print(f"line: {line}")
        if line:
            if section_regex.match(line):
                print(f"section regex match; isbreak = {isbreak}")
                # if isbreak == 0:
                #     isbreak == 1
                versenum += 1
                print(f"versenum: {versenum}")
                verses.append({'text': "", 'chords': []})
            else:
                print(f"No section regex match; isbreak = {isbreak}")
                # if isbreak == 1:
                #     isbreak == 0
                is_chord_line = True
                for word in line.split():
                    if not(is_chord_line and chord_regex.match(word)):
                        is_chord_line = False
                print(f"match: {is_chord_line}")
                if is_chord_line:
                    verses[versenum]['chords'] += get_chord_list(line)
                    print(verses[versenum]['chords'])
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

    return verses

# if __name__ == '__main__':
#     file_path = 'song.txt'
#     preprocessed_data = preprocess_song(file_path)
#     print(preprocessed_data)
# chord_regex = re.compile(r"\s*[A-G][#b]?m?(maj7|maj|min7|min|7|sus2|sus4)?")
# print(chord_regex.search("D   "))
print(preprocess_song("data/A Change Is Gonna Come  Sam Cooke.txt"))