import random
import numpy as np
import keras
from keras import layers
    

filepath = keras.utils.get_file('shakespeare.txt',
                                'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

text = text[300000:800000]

characters = sorted(set(text))

char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

SEQ_LENGTH = 40
STEP_SIZE = 3

""" Create Model For Training """
sentences = []
next_char = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i + SEQ_LENGTH])
    next_char.append(text[i + SEQ_LENGTH])
print("Number of sequences:", len(sentences))

x = np.zeros((len(sentences), SEQ_LENGTH,
              len(characters)), dtype=np.bool_)
y = np.zeros((len(sentences),
              len(characters)), dtype=np.bool_)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_char[i]]] = 1

model = keras.Sequential(
    [
        keras.Input(shape=(SEQ_LENGTH, len(characters))),
        layers.LSTM(128),
        layers.Dense(len(characters), activation="softmax"),
    ]
)
optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(loss="categorical_crossentropy", optimizer=optimizer)

model.fit(x, y, batch_size=256, epochs=4)

"""Save & Load Training Model """
# model.save('m.h5')
# model = keras.models.load_model('my_model.h5')

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ""
    sentence_text = text[start_index : start_index + SEQ_LENGTH]
    generated += sentence_text
    for i in range(length):
        x = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, character in enumerate(sentence_text):
            x[0, t, char_to_index[character]] = 1

        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence_text = sentence_text[1:] + next_character
    return generated


print('------------0.2------------')
print(generate_text(300, 0.2))
print('------------0.3------------')
print(generate_text(300, 0.3))
print('------------0.4------------')
print(generate_text(300, 0.4))
print('------------0.5------------')
print(generate_text(300, 0.5))
print('------------0.6------------')
print(generate_text(300, 0.6))
print('------------0.8------------')
print(generate_text(300, 0.8))
print('------------1.0------------')
print(generate_text(300, 1.0))
