from utils import re

def preprocess_song_sections(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Regular expressions to identify chords and tabs
    chord_regex = re.compile(r"[A-G][#b]?m?(maj7|maj|min7|min|7|sus4)?")
    tab_regex = re.compile(r"^\s*[eBGDAE]\|")
    section_regex = re.compile(r"^\[.*\]$")

    current_section = None
    song_data = {
        'tabs': {},
        'lyrics_and_chords': {}
    }

    chords = ""
    lyrics = ""

    for line in lines:
        line = line.strip()

        if section_regex.match(line):
            current_section = line.strip('[]')
            # Initialize sections in the dictionary
            song_data['lyrics_and_chords'][current_section] = []
        elif tab_regex.match(line):
            # Collect tab lines under the current section
            if 'tabs' not in song_data:
                song_data['tabs'] = {}
            if current_section not in song_data['tabs']:
                song_data['tabs'][current_section] = []
            song_data['tabs'][current_section].append(line)
        else:
            if chord_regex.search(line):
                chords = line
            else:
                lyrics = line

            if chords and lyrics:
                song_data['lyrics_and_chords'][current_section].append({'chords': chords, 'lyrics': lyrics})
                chords = ""  # Reset after pairing
                lyrics = ""

    return song_data

if __name__ == '__main__':
    file_path = 'song.txt'
    preprocessed_data = preprocess_song_sections(file_path)
    print(preprocessed_data)
