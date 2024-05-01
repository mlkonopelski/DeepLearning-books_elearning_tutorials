import torch
import yaml 
from model import BigramLanguageModel
from datetime import datetime

torch.manual_seed(1337)

BATCH_SIZE = 64
BLOCK_SIZE = 256
TOKEN_EMBEDDINGS = 250
HEADS = 5
BLOCKS = 5
DROPOUT = 0.2

MAX_ITER = 5000
EVAL_ITER = 100
LR_RATE = 3e-4 

# with open("config.yml", "r") as f:
#     config = yaml.safe_load(f)

DEVICE = 'cpu' #config['device']

start = datetime.now()
# -----------------------------
# DATA
# -----------------------------

with open('.data/tinyshakespear.txt', 'r') as f:
    text = f.read()
    
# all unique characters that appear in the text 
chars = sorted(list(set(text)))

# create mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# encode text
data = torch.tensor(encode(text))
data = data.to(DEVICE)

# train test split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE, ))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1: i+BLOCK_SIZE+1] for i in ix])
    return x,y

vocab_size = len(chars)
model = BigramLanguageModel(vocab_size, n_embed=TOKEN_EMBEDDINGS, block_size=BLOCK_SIZE, blocks=BLOCKS, heads=HEADS, dropout=DROPOUT, device=DEVICE)
model = model.to(DEVICE)

print(f'Model has {sum([p.numel() for p in model.parameters()]) / 1e6}M parameters.')

@torch.no_grad()
def eval_loss(start_time):
    
    loss_dict = {} 
    model.eval()
    for split in ['train', 'test']:
        xb, yb = get_batch(split)
        _, loss = model(xb, yb)
        loss_dict[split] = loss
    print(f"[{iter}/{MAX_ITER}] Train loss: {loss_dict['train']} | Test loss: {loss_dict['test']} in {datetime.now() - start_time}")

def validate_model():
    print(
        decode(
            model.generate(
                idx=torch.zeros((1, 1), dtype=torch.long).to(config['device']),
                max_new_tokens=256
            )[0].tolist()
        )
    )
    
optimizer = torch.optim.AdamW(params=model.parameters(), 
                              lr=LR_RATE)

for iter in range(MAX_ITER):
    
    if iter % EVAL_ITER == 0:
        eval_loss(start)
        
    xb, yb = get_batch('train')
    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
    
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"Finished (on: {DEVICE}) in {datetime.now() - start}")

validate_model()