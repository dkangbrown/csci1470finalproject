import pandas as pd
import numpy as np
import string
from sklearn.model_selection import train_test_split
from config import *
from better_profanity import profanity


def load_data():
    print('Loading data...')
    df = pd.read_csv("../../data/musicoset_songfeatures/musicoset_songfeatures/lyrics.csv", sep="\t")
    df['lyrics'] = df.apply(lambda x: np.nan if len(str(x['lyrics'])) < 10 else str(x['lyrics'])[2:-2], axis=1)
    df = df.dropna()

    pdf = pd.read_csv('../../data/poetry-foundations/PoetryFoundationData.csv', quotechar='"')
    pdf['Poem'] = pdf['Poem'].apply(lambda poem: profanity.censor(poem))
    pdf['single_text'] = pdf['Poem'].apply(lambda x: ' \n '.join([l.lower().strip().translate(TRANSLATOR) for l in x.splitlines() if len(l)>0]))

    df = df.join(df.apply(split_text, axis=1))
    sum_df = pd.DataFrame(df.iloc[:1000]['single_text'])
    sum_df = sum_df.append(pd.DataFrame(pdf.iloc[:1000]['single_text']), ignore_index=True)
    sum_df.dropna(inplace=True)

    return sum_df


def split_text(x):
    text = x['lyrics']
    text = profanity.censor(text)
    sections = text.split('\\n\\n')
    keys = {'Verse 1': np.nan, 'Verse 2': np.nan, 'Verse 3': np.nan, 'Verse 4': np.nan, 'Chorus': np.nan}
    single_text = []
    for s in sections:
        key = s[s.find('[') + 1:s.find(']')].strip()
        if ':' in key:
            key = key[:key.find(':')]
        if key in keys:
            single_text += [x.lower().replace('(','').replace(')','').translate(TRANSLATOR) for x in s[s.find(']')+1:].split('\\n') if len(x) > 1]
    res = {'single_text': ' \n '.join(single_text)}
    return pd.Series(res)


def prepare_data(data):
    print('Preparing data...')
    # Flatten all text into a single list of words
    text_as_list = [word for text in data['single_text'] for word in text.split()]
    
    # Calculate word frequencies
    frequencies = {}
    for word in text_as_list:
        if word.strip() != '' or word == '\n':
            frequencies[word] = frequencies.get(word, 0) + 1

    # Filter out uncommon words
    uncommon_words = set(word for word, freq in frequencies.items() if freq < MIN_FREQUENCY)
    words = sorted(word for word, freq in frequencies.items() if freq >= MIN_FREQUENCY)

    # Create sequences that do not include uncommon words
    valid_seqs = []
    next_words = []
    for i in range(len(text_as_list) - MIN_SEQ):
        current_seq = text_as_list[i:i + MIN_SEQ + 1]
        if not set(current_seq).intersection(uncommon_words):
            valid_seqs.append(current_seq[:-1])
            next_words.append(current_seq[-1])

    print('Total words:', len(text_as_list))
    print('Words with less than {} appearances: {}'.format(MIN_FREQUENCY, len(uncommon_words)))
    print('Words with more than {} appearances: {}'.format(MIN_FREQUENCY, len(words)))
    print('Valid sequences of size {}: {}'.format(MIN_SEQ, len(valid_seqs)))

    return valid_seqs, next_words, words
