from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my dog.',
    'I love my cat',
    'You love my dog!',
    'Do you think my cat is amazing'
]

# 1. Build Tokenizer
tokenizer = Tokenizer(100, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
index_word = tokenizer.index_word
print(index_word)

#. Encode sentences to code
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)

# padd senteces to common len
padded = pad_sequences(sequences, maxlen=10, padding='post')
print(padded)

# encode new sentences
new_text = [
    'I love cats',
    'My love for cat is love.'
]

new_text_sequence = tokenizer.texts_to_sequences(new_text)
new_text_padded = pad_sequences(new_text_sequence)
print(new_text_padded)
