from data_preparation import load_data
from model import LyricsGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import logging

MIN_FREQUENCY = 7
MIN_SEQ = 5
BATCH_SIZE = 32


def prepare_data(data):
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


def main():
    # Load data
    logging.info("Loading data...")
    data = load_data()
    logging.info("Data loaded successfully.")

    # Prepare data and vocabulary
    logging.info("Preparing data...")
    sequences, next_words, words = prepare_data(data)
    logging.info("Data prepared successfully.")

    # Split the data into training and testing sets
    logging.info("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(sequences, next_words, test_size=0.2, random_state=42)
    logging.info("Data split successfully.")

    # Initialize and build the lyrics generator model
    logging.info("Building the model...")
    lg = LyricsGenerator(words)

    # Train the model
    lg.train(X_train, y_train, X_test, y_test, BATCH_SIZE)

    # Save the final trained model
    # lg.model.save('final_lyrics_model.h5')
    # logging.info("Model saved successfully.")

    # Generate sample text to check the model's performance
    seed_text = ['hello', 'from', 'the', 'other', 'side']
    print("Sample generated text:", lg.generate_text(seed_text, num_words=50))

if __name__ == "__main__":
    main()

