from utils import argparse, sys
from tensorflow.keras.models import load_model as keras_load_model
from lyrics.model import LyricsGenerator

# Default paths to models
LYRICS_MODEL_PATH = "lyrics/lyrics_generator.h5"
CHORDS_MODEL_PATH = "path_to_chords_model"
SENTIMENT_MODEL_PATH = "sentiment/sentiment.keras"


def load_model(model_path):
    print(f"Loading model from {model_path}")
    return keras_load_model(model_path)


def generate_lyrics(model, seed):
    print(f"Generating lyrics using model at {model} with seed: '{seed}'")
    lyrics_generator = LyricsGenerator(seed, model)
    return lyrics_generator.generate_text(seed, diversity=0.5)


def generate_chords(model, lyrics):
    print(f"Generating chords using model at {model} for lyrics: '{lyrics}'")
    return "Generated chords"  # Placeholder for chord generation


def main():
    parser = argparse.ArgumentParser(description="Generate lyrics and chords from a seed.")
    parser.add_argument("seed", help="Input seed for generating lyrics.")
    parser.add_argument("--lyrics_only", action="store_true", help="Only generate lyrics.")
    parser.add_argument("--chords_only", help="Only generate chords from the provided lyrics text.")
    parser.add_argument("--lyrics_model_path", default=LYRICS_MODEL_PATH, help="Path to the lyrics model file.")
    parser.add_argument("--chords_model_path", default=CHORDS_MODEL_PATH, help="Path to the chords model file.")

    args = parser.parse_args()

    if args.chords_only:
        chords_model = load_model(args.chords_model_path)
        chords = generate_chords(chords_model, args.chords_only)
        print(f"Generated Chords:\n{chords}")
    else:
        lyrics_model = load_model(args.lyrics_model_path)
        lyrics = generate_lyrics(lyrics_model, args.seed)

        if args.lyrics_only:
            print(f"Generated Lyrics:\n{lyrics}")
        else:
            chords_model = load_model(args.chords_model_path)
            chords = generate_chords(chords_model, lyrics)
            print(f"Generated Lyrics:\n{lyrics}")
            print(f"Generated Chords:\n{chords}")


if __name__ == "__main__":
    main()
