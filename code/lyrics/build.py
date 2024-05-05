import argparse
from model import LyricsGenerator


def main(retrain):
    generator = LyricsGenerator(retrain=retrain)

    if not args.text:
        seed_text = "Never gonna give you up"
    else:
        seed_text = args.text

    # Generate different diversity levels
    for diversity in [0.2, 0.5, 1.0]:
        print(f"Generated text with diversity {diversity}:", generator.generate_text(seed_text, diversity=diversity))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage Lyrics Generator training and prediction.")
    parser.add_argument("--retrain", action="store_true", help="Set this flag to retrain the model.")
    parser.add_argument("--text", type=str, help="Seed text to generate lyrics.")
    args = parser.parse_args()
    main(args.retrain)
