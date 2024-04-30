from utils import re

chord_regex = re.compile(r"^[A-G][#b]?m?(7|5|M|maj7|maj|M7|min7|m7|min|dim|aug|\+|sus2|sus4|7sus2|7sus4)?(add)?[0-9]*/?[A-G]?$")
if chord_regex.match('C+999/F'):
    print('match!')
else:
    print('not match!')