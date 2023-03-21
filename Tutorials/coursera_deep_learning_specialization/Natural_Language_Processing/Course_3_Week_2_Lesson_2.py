import json
import urllib.request
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


def download_data(json_path):
    with urllib.request.urlopen(json_path) as url:
        data = json.loads(url.read().decode())

    sentences = []
    labels = []

    for item in data:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])

    labels = np.array(labels)

    return sentences, labels


def prepare_datasets(sentences, labels, train_split, vocab_size, oov_token, max_len):
    train_sentences = sentences[:train_split]
    train_labels = labels[:train_split]

    test_sentences = sentences[train_split:]
    test_labels = labels[train_split:]

    tokenizer = Tokenizer(vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(train_sentences)

    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    train_p_sequences = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')

    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    test_p_sequences = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')

    return ((train_p_sequences, train_labels), (test_p_sequences, test_labels))


def train_model(data, vocab_size, embed_dim, max_len):

    (train_p_sequences, train_labels), (test_p_sequences, test_labels) = data

    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_len),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_p_sequences, train_labels,
                        epochs=20,
                        validation_data=(test_p_sequences, test_labels),
                        verbose=0)

    return history


if __name__ == '__main__':
    json_path = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json'

    TRAIN_SPLIT = 20000
    OOV_TOKEN = '<OOV>'
    VOCAB_SIZE = 5000
    MAX_LEN = 250
    EMBED_DIM = 25


    sentences, labels = download_data(json_path)
    data = prepare_datasets(sentences, labels, TRAIN_SPLIT, VOCAB_SIZE, OOV_TOKEN, MAX_LEN)
    history = train_model(data, VOCAB_SIZE, EMBED_DIM, MAX_LEN)

    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")
