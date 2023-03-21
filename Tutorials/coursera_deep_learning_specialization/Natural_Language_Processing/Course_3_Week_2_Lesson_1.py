import io
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

NUM_WORDS = 10000
OOV_TOKEN = '<OOV>'
MAX_LEN = 120
EMB_DIMS = 16
DATA_PATH = 'C:/Users/Lenovo/PycharmProjects/DeepLearning/data/'


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


data, desc = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

train, test = data['train'], data['test']

train_texts = []
train_labels = []
test_texts = []
test_labels = []

for t, l in train:
    train_texts.append(t.numpy().decode('utf8'))
    train_labels.append(l.numpy())

for t, l in train:
    test_texts.append(t.numpy().decode('utf8'))
    test_labels.append(l.numpy())

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

tokenizer = Tokenizer(NUM_WORDS, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(train_texts)
train_sentences = tokenizer.texts_to_sequences(train_texts)
train_p_sentences = pad_sequences(train_sentences, MAX_LEN, padding='post', truncating='post')

test_sentences = tokenizer.texts_to_sequences(test_texts)
test_p_senteces = pad_sequences(test_sentences, MAX_LEN, padding='post', truncating='post')

word_index = tokenizer.word_index
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(NUM_WORDS, EMB_DIMS, input_length=MAX_LEN),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(60, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_p_sentences,
                    train_labels,
                    epochs=10,
                    validation_data=(test_p_senteces, test_labels))

model.save(DATA_PATH + 'Embedding_on_imbd_review.h5')

weights = model.layers[0].get_weights()[0]

out_v = io.open(DATA_PATH + 'imbd_reviews_vecs.tsv', 'w', encoding='utf-8')
out_m = io.open(DATA_PATH + 'imbd_reviews_meta.tsv', 'w', encoding='utf-8')

for word_ix in range(1, NUM_WORDS):
    word = reverse_word_index[word_ix]
    weight = weights[word_ix]
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in weight]) + "\n")

out_v.close()
out_m.close()
