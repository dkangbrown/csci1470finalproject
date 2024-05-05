import argparse
from model import LyricsGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate text with a LyricsGenerator model.")
    parser.add_argument('--retrain', action='store_true', help='Retrain the model instead of using the saved one.')
    parser.add_argument('--text', type=str, default="Never gonna give you up", help='Seed text to start generating from.')
    args = parser.parse_args()

    # Initialize the LyricsGenerator with the retrain flag
    generator = LyricsGenerator(retrain=args.retrain)

    # Generate different diversity levels
    for diversity in [0.2, 0.5, 1.0]:
        print(f"----- Diversity: {diversity} -----")
        generated_text = generator.generate_text(args.text, diversity=diversity, num_words=50)
        for line in generated_text:
            print(line)
        print('='*80)

if __name__ == "__main__":
    main()
