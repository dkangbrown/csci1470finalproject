from utils import re

#(maj7|maj|min7|min|7|sus2|sus4)?
chord_regex = re.compile(r"^[A-G][#b]?m?(maj7|maj|min7|min|7|sus2|sus4)?$")
section_regex = re.compile(r"^(\[|#)?(chorus|Chorus|CHORUS|verse|Verse|VERSE|intro|Intro|INTRO|outro|Outro|OUTRO|bridge|Bridge|BRIDGE|interlude|Interlude|INTERLUDE|instrumental|Instrumental|INSTRUMENTAL|solo|Solo|SOLO)*( )?[0-9]*(\]|\:|\.)?$")
accidental_regex = re.compile(r"^[#b]$")

def is_chord(chord):
    if chord_regex.match(chord):
        print('match!')
    else:
        print('not match :(')

def is_section(chord):
    if section_regex.match(chord):
        print('match!')
    else:
        print('not match :(')

is_chord('A')
is_chord('A#m')
is_chord('Abm7')
is_chord('Absus4')
is_chord('Amaj7')

is_section('[chorus]')
is_section('[Chorus]')
is_section('chorus.')
is_section('#1.')
is_section('verse 1.')
is_section('VERSE1:')
is_section('Verse 1')
is_section('dlkfjalfkdfd')
is_section('love')
is_section('[love]')

qual = 'Cmaj7add9'
if re.compile(r"^.*(add|6|8|9|11|13|15|17|19)$").match(qual):
    qual = qual[0:min(qual.index('add'),qual.index('6'),qual.index('8'),qual.index('9'),qual.index('11'),qual.index('13'),qual.index('15'),qual.index('17'),qual.index('19'))]

print(qual)