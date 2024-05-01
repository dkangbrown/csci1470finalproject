import pandas as pd
import numpy as np
import string

TRANSLATOR = str.maketrans('', '', string.punctuation)

def load_data():
    df = pd.read_csv("../../data/musicoset_songfeatures/musicoset_songfeatures/lyrics.csv", sep="\t")
    df['lyrics'] = df.apply(lambda x: np.nan if len(str(x['lyrics'])) < 10 else str(x['lyrics'])[2:-2], axis=1)
    df = df.dropna()

    pdf = pd.read_csv('../../data/poetry-foundations/PoetryFoundationData.csv', quotechar='"')
    pdf['single_text'] = pdf['Poem'].apply(lambda x: ' \n '.join([l.lower().strip().translate(TRANSLATOR) for l in x.splitlines() if len(l)>0]))

    df = df.join(df.apply(split_text, axis=1))
    sum_df = pd.DataFrame(df.iloc[:1000]['single_text'])
    sum_df = sum_df._append(pd.DataFrame(pdf.iloc[:1000]['single_text']), ignore_index=True)
    sum_df.dropna(inplace=True)

    return sum_df


def split_text(x):
    text = x['lyrics']
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

