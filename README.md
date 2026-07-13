# neural-networks-gen-text

Recurrent neural networks are very useful when it comes to processing sequential data like text. This project uses an LSTM (Long Short-Term Memory) network to teach a computer to write text in the style of Shakespeare, learning character by character rather than word by word.

The model is trained on a 500,000-character slice of Shakespeare's collected works and generates new text one character at a time, with adjustable "temperature" to control how creative or conservative the output is.

## How It Works

The model doesn't understand words or grammar. It learns statistical patterns between characters by looking at 40-character windows of text and predicting what character comes next. Given enough training, it picks up on spelling, common words, punctuation patterns, and even loose sentence structure, purely from character sequences.

Generation is autoregressive: the model predicts one character, appends it to the input, drops the oldest character, and repeats. A temperature parameter reshapes the prediction probabilities before sampling, letting you trade off between safe, repetitive text and more varied, riskier text.

## Features

- Downloads and caches the Shakespeare corpus automatically on first run
- Character-level one-hot encoding pipeline built from scratch with NumPy
- Single-layer LSTM model trained with Keras
- Temperature-based sampling function for controllable text generation
- Generates sample text at seven temperature levels (0.2 to 1.0) for side-by-side comparison

## Requirements

```
python >= 3.8
tensorflow / keras
numpy
```

Install dependencies:

```bash
pip install tensorflow numpy
```

## Usage

Run the script directly:

```bash
python shakespeare_generator.py
```

On first run, it will:

1. Download `shakespeare.txt` from Google's TensorFlow storage bucket
2. Preprocess the text and build character-to-index mappings
3. Build training sequences using a sliding window
4. Train an LSTM for 4 epochs
5. Generate and print 300-character samples at temperatures `0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0`

## Model Architecture

| Layer | Output Shape | Details |
|---|---|---|
| Input | `(40, vocab_size)` | One-hot encoded character sequence |
| LSTM | `(128,)` | 128 hidden units |
| Dense | `(vocab_size,)` | Softmax over all characters |

**Training configuration:**

| Parameter | Value |
|---|---|
| Sequence length | 40 characters |
| Step size | 3 characters |
| Batch size | 256 |
| Epochs | 4 |
| Optimizer | RMSprop (lr = 0.01) |
| Loss | Categorical crossentropy |

## Understanding Temperature

The `sample()` function reshapes the softmax output before drawing a character:

| Temperature | Behavior |
|---|---|
| 0.2 – 0.4 | Conservative, repetitive, closer to the most likely next character |
| 0.5 – 0.6 | Balanced mix of coherence and variety |
| 0.8 – 1.0 | More random and creative, higher chance of gibberish |

Lower values sharpen the probability distribution toward the model's top prediction. Higher values flatten it, giving less likely characters a better shot at being picked.

## Example Output

Output quality depends heavily on training time. With only 4 epochs, expect rough, partially-formed English rather than fully coherent Shakespeare:

```
------------0.2------------
the king, and the strange the more the strong the state
the strength of the state and the strange the more the...

------------1.0------------
wor'd svle, ay honou-t, wich? shall dremp and pxow's br...
```

## Saving and Loading the Model

Model saving is included but commented out by default:

```python
model.save('m.h5')
model = keras.models.load_model('my_model.h5')
```

Uncomment these lines to persist the trained model and skip retraining on future runs. Note that `.h5` is Keras's legacy format; newer Keras versions recommend the `.keras` extension instead.

## Known Limitations

- **Slow generation**: each character requires a separate `model.predict()` call, so generating long passages at multiple temperatures can take a while.
- **Short training run**: 4 epochs is enough to see the model start learning structure, but not enough for fully fluent output. Increase epochs for better results.
- **Manual one-hot encoding**: works fine for a small character vocabulary, but doesn't scale to word-level vocabularies or very long sequences. An `Embedding` layer would be a natural upgrade.
- **Fixed corpus slice**: only characters 300,000–800,000 of the source file are used, skipping the license header and trimming dataset size for faster iteration.

## Ideas for Extending This Project

- Replace the one-hot input with a trainable `Embedding` layer
- Stack additional LSTM or GRU layers, or add dropout for regularization
- Train for more epochs and track loss over time
- Swap the character-level LSTM for a small transformer for higher-quality generation
- Add a CLI or simple web interface for interactive generation with adjustable temperature and length

## Credits

Trained on the Shakespeare dataset hosted by TensorFlow at:
`https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt`
