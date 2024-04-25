# cs2470-final-project

## Generation of Chords from Song Lyrics

### Name and Logins
Bisheshank C. Aryal: bcaryal Ryan Lee: xlee4 Daniel Kang: dkang33 Chai Harsha: charsha

### Introduction

We will develop a model that will generate chords based on a text given as input. At the most abstract level, this is a form of sentiment analysis, where a chord progression is a representation of the “feeling” or the “message” of a given text.

We came up with this idea by starting with the idea of chord prediction paralleling word prediction. As our understanding of sentence generation models grew, we were able to imagine more complicated modes of chord generation, and the most ambitious project we can think of at the moment is to use encoders and decoders to convert between text and chords.

This is a structured prediction task paralleling a language translation text. Instead of translating from English to French, we are translating from English to chords (albeit chords don’t capture all of the sentence’s meaning).

### Related Work

List sing chords: AI Predicts Musical Genre by Identifying Surprising Patterns in Hit Songs https://viterbischool.usc.edu/news/2019/08/ai-predicts-musical-genre-by-identifying-surprising-patterns-in-hit-songs/ LSTM: ChordSuggester: using LSTMs to predict musical chord https://medium.com/@huanlui/chordsuggester-i-3a1261d4ea9e LSTM: Predicting Music Using Machine Learning https://link.springer.com/chapter/10.1007/978-3-031-37649-8_3 Stanford course: Machine Learning in Automatic Music Chords Generation https://cs229.stanford.edu/proj2015/136_report.pdf More music theory heavy: Combining Real-Time Extraction and Prediction of Musical Chord Progressions for Creative Applications https://www.mdpi.com/2079-9292/10/21/2634 Sound to Harmony: Harmonic Classification with Enhancing Music Using Deep Learning Techniques https://www.hindawi.com/journals/complexity/2021/5590996/

Implementations https://github.com/usc-sail/media-eval-2020

### Summary of “AI Predicts Musical Genre by Identifying Surprising Patterns in Hit Songs” 
This model won the “2020-Emotion-and-Theme-Recognition-in-Music-Task”. It did so by aiming to focus on letting AI discover complicated patterns between lyrics and chords, instead of relying on all kinds of input an audio file can provide. Thus, the model trained solely on text data and chords, and especially on the relationship between them, instead of the audio file that would show all aspects of the song, including the instrumentation and tempo. Unraveling the interaction between lyrics and chords was revolutionary, first because it is relatively light-weight to train, and also because it seems to be what often characterizes different genres. For example, the hit song “Old Town Road” was predicted to have the lyrics of a country song, the chord progression of a rock song, but the combined effect of a pop song. Such detailed classification allowed the model to win the competition. The architecture used was a CNN architecture with some residual streams.

### Data

https://guitarresource.net/2020/11/11/chord-and-lyric-text-txt-file-library/ The data has 500 songs, each with multiple verses. To get more songs we could go into the following website and copy paste chords paired with lyrics (almost unlimited size): https://www.chordie.com/ More databases here used by competition winners: https://multimediaeval.github.io/2020-Emotion-and-Theme-Recognition-in-Music-Task/results

There are a lot of preprocessing needed, here is a list of preprocessing needed: Getting rid of extra labels (verse labels, etc.) Connecting chords to words Normalizing chords to a predetermined tonic / making chord progressions relative Dividing songs into “progressions / phrases” Preprocessing text

### Methodology
The model will be a transformer model. The architecture works best because we need to convert between two different mediums (English and chord). Now, we use transformers rather than RNNs to make a more efficient model to train with parallel computing. Following are ideas for training Calculate loss for each individual chord prediction (chord difference can be measured by the cycle of fifths and chord quality) Calculate loss for the entire chord progression (progression difference based on cadence and other aspects) If this does not work, we can try changing our measure of loss, or divide the task to 1. sentiment analysis using bag of words 2. chord generation

### Metrics

Accuracy metric: predicts the correct ratio of minor chords, diminished chords, major chords in an entire verse Using a strict accuracy metric is not good, for different chords have differing similarity to each other (perfect fourth chord can function similarly to a minor second chord), and there are many solutions to resolve a particular chord progression, which means there is less of a “correct answer” to which another answer differs by a specific amount. Base goal: Get the model to produce sensible progressions (with cadence), and get the general minor vs major division correct Target goal: Have the model be somewhat sensitive to the words above the sentiment analysis (happy and sad). To do this, we look at the ratio of different chord qualities. Stretch goal: For lyrics of similar topics, similar progressions are predicted (same cadence, same chord quality).

### Ethics

Deep learning is a good approach for this task for the same reason it is a good way to perform machine translation. There are many variables to how lyrics are composed, and to how chord progressions are structured. Both the text and the chords are not composed deterministically—there are creative aspects to them, and thus a deterministic program cannot capture their relationship.

I will talk about copyright issues. If a model successfully performs this task at a level that matches basic level composition, more people can start making songs more easily using the text they write. The user gets more control over what chords are generated than existing models that do not take lyrics into account. This means, a chord progression created with this tool has an ambiguous status whether it belongs to the lyric writer or not. If they compose a song this way, we may run into copyright issues, of which the stakeholders are: the lyric writer, the singer, the model creator, and potentially the person who made a progression that the model’s output resembles. The last stakeholder is crucial, for they are the source of data from which the model was trained. Thus, it is likely that the model will output something similar to what already exists out there. Whether this constitutes plagiarism is a complicated ethical question.

### Division of Labor

Each person will take one of each category. It is not determined yet, but an arbitrary order would be: Bisheshank, Chai, Daniel, Ryan, respectively. Preprocessing Data Getting data together, into the correct format Parsing the data Getting chords to show relative, rather than absolute qualities Preprocessing text Model Creation Creating a loss model Baseline transformer architecture Training Debugging
