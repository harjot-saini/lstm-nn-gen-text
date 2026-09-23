"""
LSTM Character-Level Text Generator
------------------------------------
Trains a character-level LSTM on Shakespeare's works and generates new text
at several sampling temperatures.

Usage:
    python lstm_text_generator.py --epochs 4 --seq-length 40
    python lstm_text_generator.py --load-model model.keras --skip-training
"""
from __future__ import annotations

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import keras
from keras import layers



logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
GET_DATA = False
DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
TEXT_SLICE = slice(300_000, 800_000)  # keep the run fast; full text works too


def load_text(url: str = DATA_URL, text_slice: slice = TEXT_SLICE) -> str:
    """Download (or reuse the cached copy of) the training corpus and lowercase it."""
    file_path = "shakespeare.txt"
    if GET_DATA :
        file_path = keras.utils.get_file(file_path, url)
    text = Path(file_path).read_text(encoding="utf-8").lower()
    return text[text_slice]


def build_vocab(text: str) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """Build the character vocabulary and its index mappings."""
    characters = sorted(set(text))
    char_to_index = {c: i for i, c in enumerate(characters)}
    index_to_char = {i: c for i, c in enumerate(characters)}
    return characters, char_to_index, index_to_char


def vectorize_text(
    text: str,
    characters: list[str],
    char_to_index: dict[str, int],
    seq_length: int,
    step_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window over the text to build (sequence -> next character) training pairs."""
    sentences, next_char = [], []
    for i in range(0, len(text) - seq_length, step_size):
        sentences.append(text[i : i + seq_length])
        next_char.append(text[i + seq_length])
    logger.info("Number of training sequences: %d", len(sentences))

    x = np.zeros((len(sentences), seq_length, len(characters)), dtype=np.float32)
    y = np.zeros((len(sentences), len(characters)), dtype=np.float32)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_to_index[char]] = 1.0
        y[i, char_to_index[next_char[i]]] = 1.0
    return x, y


def build_model(seq_length: int, vocab_size: int, learning_rate: float = 0.01) -> keras.Model:
    """Define and compile a single-layer LSTM character predictor."""
    model = keras.Sequential(
        [
            keras.Input(shape=(seq_length, vocab_size)),
            layers.LSTM(128),
            layers.Dense(vocab_size, activation="softmax"),
        ]
    )
    optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)
    return model


def sample(preds: np.ndarray, temperature: float = 1.0, epsilon: float = 1e-8) -> int:
    """Sample a character index from a probability distribution at the given temperature."""
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + epsilon) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return int(np.argmax(probas))


def generate_text(
    model: keras.Model,
    text: str,
    characters: list[str],
    char_to_index: dict[str, int],
    index_to_char: dict[int, str],
    seq_length: int,
    length: int,
    temperature: float,
) -> str:
    """Seed the model with a random real excerpt and generate `length` new characters."""
    start_index = random.randint(0, len(text) - seq_length - 1)
    sentence_text = text[start_index : start_index + seq_length]
    generated = sentence_text

    for _ in range(length):
        x = np.zeros((1, seq_length, len(characters)), dtype=np.float32)
        for t, character in enumerate(sentence_text):
            x[0, t, char_to_index[character]] = 1.0

        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence_text = sentence_text[1:] + next_character

    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train (or load) an LSTM text generator on Shakespeare.")
    parser.add_argument("--seq-length", type=int, default=40, help="Length of input character sequences.")
    parser.add_argument("--step-size", type=int, default=3, help="Stride between training sequences.")
    parser.add_argument("--epochs", type=int, default=4, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="RMSprop learning rate.")
    parser.add_argument("--gen-length", type=int, default=300, help="Number of characters to generate per sample.")
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
        help="Sampling temperatures to showcase.",
    )
    parser.add_argument("--save-model", type=str, default=None, help="Path to save the trained model to, e.g. model.keras")
    parser.add_argument("--load-model", type=str, default=None, help="Path to a saved model to load instead of training.")
    parser.add_argument("--skip-training", action="store_true", help="Skip training (requires --load-model).")
    parser.add_argument(
        "--output-file",
        type=str,
        default="output.txt",
        help="Where to save the generated samples (set to '' to skip saving).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("Loading and preprocessing text corpus...")
    text = load_text()
    characters, char_to_index, index_to_char = build_vocab(text)

    if args.load_model:
        logger.info("Loading model from %s", args.load_model)
        model = keras.models.load_model(args.load_model)
    else:
        model = build_model(args.seq_length, len(characters), args.learning_rate)

    if not args.skip_training and not args.load_model:
        x, y = vectorize_text(text, characters, char_to_index, args.seq_length, args.step_size)
        logger.info("Training for %d epochs...", args.epochs)
        model.fit(x, y, batch_size=args.batch_size, epochs=args.epochs)

        if args.save_model:
            model.save(args.save_model)
            logger.info("Model saved to %s", args.save_model)

    output_blocks = []
    for temperature in args.temperatures:
        header = f"\n------------ temperature={temperature} ------------"
        sample_text = generate_text(
            model,
            text,
            characters,
            char_to_index,
            index_to_char,
            args.seq_length,
            args.gen_length,
            temperature,
        )
        print(header)
        print(sample_text)
        output_blocks.append(f"{header}\n{sample_text}")

    if args.output_file:
        Path(args.output_file).write_text("\n".join(output_blocks) + "\n", encoding="utf-8")
        logger.info("Generated samples saved to %s", args.output_file)


if __name__ == "__main__":
    main()