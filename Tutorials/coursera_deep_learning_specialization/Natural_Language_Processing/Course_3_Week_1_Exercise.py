import requests
import csv

path = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/bbc-text.csv'
csv_path = 'C:/Users/Lenovo/PycharmProjects/DeepLearning/data/bbc-text.csv'
stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]


bbc_text_raw = requests.get(path).content
csv_file = open(csv_path, 'wb')
csv_file.write(bbc_text_raw)
csv_file.close()

sentences = []
labels = []

with open(csv_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        labels.append(row[0])
        sentence = ' '.join([word for word in row[1].split(' ') if word.lower() not in stopwords])
        sentences.append(sentence)

print(len(sentences))
print(sentences[0])
print(labels[0])

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

print(len(word_index))

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences)

print(padded[0])
print(padded.shape)

tokenizer2 = Tokenizer()
tokenizer2.fit_on_texts(labels)
label_word_index = tokenizer2.word_index

label_seq = tokenizer2.texts_to_sequences(labels)

print(label_seq)
print(label_word_index)
