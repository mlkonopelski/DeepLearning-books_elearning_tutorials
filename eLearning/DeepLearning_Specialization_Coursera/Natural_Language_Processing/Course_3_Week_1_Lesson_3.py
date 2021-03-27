import urllib.request
import json


path = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json'

with urllib.request.urlopen(path) as url:
    sarcasm_json = json.loads(url.read().decode())

sentences = []
labels = []
urls = []

for item in sarcasm_json:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
print(len(word_index))
print(word_index)

sequences = tokenizer.texts_to_sequences(sentences)
sequences = pad_sequences(sequences, padding='post')
print(sequences[0])
print(sequences.shape)
