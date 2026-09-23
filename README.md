# neural-networks-gen-text

A character-level text generator built with an LSTM in Keras. Trains on Shakespeare's collected works and generates new Shakespeare-style text at several sampling temperatures, from conservative and repetitive (low temperature) to creative and chaotic (high temperature).

## How It Works

1. Downloads and lowercases the [Tiny Shakespeare corpus](https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt) (a 500k-character slice is used by default to keep training fast)
2. Slides a window across the text to build `(40-character sequence → next character)` training pairs
3. One-hot encodes sequences and trains a single-layer LSTM (128 units) with a softmax output over the character vocabulary
4. Generates new text autoregressively: predict the next character, sample it, append it, slide the window forward, repeat
5. Repeats generation at multiple temperatures so you can compare how sampling randomness affects output quality

## Requirements

- Python 3.9+
- [Keras](https://keras.io/) 3 with a backend (TensorFlow is the default and simplest choice)
- NumPy

```bash
pip install keras tensorflow numpy
```

## Usage

Train and generate with default settings:
```bash
python model.py
```

This trains for 4 epochs on 40-character sequences, then generates 300-character samples at temperatures `0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0`, printing them to the console and saving them to `output.txt`.

### Useful flags

| Flag | Default | Meaning |
|---|---|---|
| `--seq-length` | `40` | Input sequence length (characters) |
| `--step-size` | `3` | Stride between training sequences |
| `--epochs` | `4` | Training epochs |
| `--batch-size` | `256` | Training batch size |
| `--learning-rate` | `0.01` | RMSprop learning rate |
| `--gen-length` | `300` | Characters generated per sample |
| `--temperatures` | `0.2 0.3 0.4 0.5 0.6 0.8 1.0` | Sampling temperatures to showcase |
| `--save-model` | `None` | Path to save the trained model (e.g. `model.keras`) |
| `--load-model` | `None` | Path to load a previously saved model instead of training |
| `--skip-training` | `False` | Skip training (pair with `--load-model`) |
| `--output-file` | `output.txt` | Where generated samples are written (`""` to skip saving) |

Full option list:
```bash
python model.py --help
```

### Examples

Quick smoke test:
```bash
python model.py --epochs 1 --gen-length 100
```

Train, save the model, and reuse it later without retraining:
```bash
python model.py --save-model model.keras
python model.py --load-model model.keras --skip-training
```

Sample at just a couple of temperatures:
```bash
python model.py --temperatures 0.4 0.8
```

## Understanding Temperature

Temperature controls how "risky" the model's character choices are:
- **Low (0.2–0.4)** — mostly picks the highest-probability character; output is repetitive but grammatically safer
- **Medium (0.5–0.6)** — a balance of coherence and variety
- **High (0.8–1.0)** — more randomness and novel word-like patterns, at the cost of coherence

Sample output from a full run is saved to `output.txt` after each execution.

## Notes

- The script uses `keras.utils.get_file`, which caches the corpus locally after the first download — subsequent runs won't re-download it.
- All one-hot tensors use `float32` rather than `bool` to avoid deprecation warnings on newer NumPy versions.
- Sampling includes a small epsilon inside the `log()` call to avoid `-inf` errors when a predicted probability is exactly 0.

## License

See [LICENSE](LICENSE) if present in this repository, otherwise treat as unlicensed / all rights reserved by the author.