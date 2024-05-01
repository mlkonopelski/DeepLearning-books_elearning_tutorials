import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

class Head(nn.Module):
    def __init__(self, C: int, T: int, head_size: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Linear(C, head_size, bias=False)
        self.key = nn.Linear(C, head_size, bias=False)
        self.value = nn.Linear(C, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(T, T)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)   # (B, T, head_size)
        k = self.key(x)     # (B, T, head_size)
        
        wei = q @ k.transpose(-2, -1)     # MatMul (B, T, T)
        wei = wei * C**-0.5     # Scale
        wei = wei.masked_fill(mask = self.tril[:T, :T] == 0, value=float('-inf'))   # masked (Opt.)
        wei = F.softmax(wei, dim=-1)        # Softmax
        wei = self.dropout(wei )
        
        v = self.value(x)

        return wei @ v
    
class MultiHead(nn.Module):
    def __init__(self, C: int, T: int, heads: int, head_size: int, dropout: float) -> None:
        super().__init__() 
        self.heads = nn.ModuleList([Head(C, T, head_size, dropout) for _ in range(heads)])
        self.projection = nn.Linear(C, C)
        self.dropout = nn.Dropout(dropout)
         
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out

 
class FeedForward(nn.Module):
    def __init__(self, C: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(C, 4 * C),
                                 nn.ReLU(),
                                 nn.Linear(4 * C, C),
                                 nn.Dropout(dropout))
    def forward(self, x):
        return self.net(x)
         

class Block(nn.Module):
    def __init__(self, C, T, heads, dropout: float) -> None:
        super().__init__()
        head_size = C // heads
        self.sa_head = MultiHead(C=C, T=T, heads=heads,head_size=head_size, dropout=dropout)
        self.ffwd = FeedForward(C, dropout)
        self.ln1 = nn.LayerNorm(C)
        self.ln2 = nn.LayerNorm(C)
        
    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x 

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embed = 32, block_size = 8, heads = 4, blocks = 3, dropout=0.2, device = 'cpu') -> None:
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.token_position_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[
            Block(n_embed, block_size, heads, dropout) for _ in range(blocks)
        ]) 
        self.l_norm = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.device = device
        
    def forward(self, idx, targets=None):
        B, T = idx.shape # batch_size, block_size
        tok_embed = self.token_embedding_table(idx) #(B, T, C)
        pos_embed = self.token_position_table(torch.arange(T, device=self.device))
        x = tok_embed + pos_embed 
        x = self.blocks(x)
        x = self.l_norm(x)
        logits = self.lm_head(x)  # (B, T, n_embed)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(-1))
                
        return logits, loss      

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current location
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]
            # apply softmax to the probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distibution
            idx_next = torch.multinomial(probs, num_samples=1)
            # add sampled idx to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    

if __name__ == '__main__':
     

    ##############################################
    # HEAD
    ##############################################
    # Visualize Head
    torch.manual_seed(1337)
    B, T, C = 4, 3, 32
    x = torch.randn(B, T, C)

    # let's see a single Head peform self-attention
    head_size = 16
    key = nn.Linear(C, head_size, bias=False)
    query = nn.Linear(C, head_size, bias=False)
    value = nn.Linear(C, head_size, bias=False)
    k = key(x)
    q = query(x)
    v = value(x)
    wei = q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) ---> (B, T, T)
    
    # add 0 mask so past tokens don't see future
    tril = torch.tril(torch.ones(T, T))
    wei = wei.masked_fill(tril==0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    out = wei @ v
