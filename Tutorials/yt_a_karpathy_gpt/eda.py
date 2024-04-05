import torch
import yaml 
from model import BigramLanguageModel

torch.manual_seed(1337)

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

with open('.data/tinyshakespear.txt', 'r') as f:
    text = f.read()
    
print(text[:1000])

# all unique characters that appear in the text 
chars = sorted(list(set(text)))
print(''.join(chars))

 # create mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode('hii there.'))
print(decode(encode('hii there.')))

data = torch.tensor(encode(text))
data = data.to(config['device'])

print(data[:1000])

# TRAIN TEST SPLIT
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
train_data[:block_size+1]

batch_size = 4

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
    return x,y

xb ,yb = get_batch('train')
print(xb.shape, yb.shape)
assert xb.shape == (batch_size, block_size)

vocab_size = len(chars)
print(vocab_size)

model = BigramLanguageModel(vocab_size)
model = model.to(config['device'])
logits, loss = model(xb, yb)


# # Example how generator works
# print(
#     decode(
#         model.generate(
#             idx=torch.zeros((1, 1), dtype=torch.long).to(config['device']),
#             max_new_tokens=100
#         )[0].tolist()
#     )
# )

# training loop



