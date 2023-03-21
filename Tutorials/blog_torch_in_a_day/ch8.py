from utils.dataset import CustomDataset
from utils.training import train
from torch.utils.data import DataLoader
from pathlib import Path

from torch import nn
import torch
from torchvision import transforms as T
import torchvision

import matplotlib.pyplot as plt
import torchinfo

#----------------------------------
#             DEVICE
#----------------------------------

DEVICE = 'mps'

#----------------------------------
#             DATA
#----------------------------------

PATH = Path('data') / 'pizza_steak_sushi'
IMG_SIZE = 224
BATCH_SIZE = 32

train_transform = T.Compose([
    T.TrivialAugmentWide(num_magnitude_bins=31),
    T.Resize(size=(IMG_SIZE, IMG_SIZE)),
    T.ToTensor()
])
    
test_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor()   
])

train_dataset = CustomDataset(path=PATH / 'train',
                              transform_input=train_transform)

test_dataset = CustomDataset(path=PATH / 'test',
                             transform_input=test_transform)


train_dataloader = DataLoader(train_dataset,  
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(test_dataset,  
                             batch_size=BATCH_SIZE,
                            shuffle=True)

# verify dataloaders:
img_batch, labels_batch = next(iter(train_dataloader))

img, label_tensor = img_batch[0], labels_batch[0]
label = train_dataset.classes[label_tensor.item()]
plt.imshow(img.permute(1, 2, 0))
plt.title(f'{label} ({img.size()})')
plt.axis = False

#----------------------------------
#             MODEL
#----------------------------------

LAYERS = 12
EMBED_SIZE = 768
MLP_SIZE = 3072
HEADS = 12
PATCH_RES = 16


class PatchEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_patches = (IMG_SIZE * IMG_SIZE) // PATCH_RES**2
        self.class_embed = nn.Parameter(data=torch.ones((1, 1, EMBED_SIZE)), requires_grad=True)
        self.position_embed = nn.Parameter(data=torch.ones((1, self.num_patches + 1, EMBED_SIZE)), requires_grad=True)
        self.linear_projection = nn.Conv2d(in_channels=3,
                      out_channels=EMBED_SIZE,
                      kernel_size=PATCH_RES,
                      stride=PATCH_RES,
                      padding=0)
        self.flatten_img = nn.Flatten(start_dim=2, end_dim=3)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        image_resolution = X.shape[-1]
        assert image_resolution % PATCH_RES == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {PATCH_RES}"
        
        # patch embedding block
        batch_size = X.shape[0]
        X = self.linear_projection(X)
        X = self.flatten_img(X)
        X = X.permute((0, 2, 1))
        X = torch.cat((self.class_embed.expand(batch_size, -1, -1), X), dim=1)
        X = X + self.position_embed
        
        return X

    
class MultiheadSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=EMBED_SIZE)
        self.msa = nn.MultiheadAttention(embed_dim=EMBED_SIZE, 
                                         num_heads=HEADS,
                                         dropout=False,
                                         batch_first=True)
    def forward(self, X):
        X = self.ln(X)
        X, _ = self.msa(query=X, key=X, value=X, need_weights=False)
        return X
    
    
class MultiLayerPerceptron(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=EMBED_SIZE)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=EMBED_SIZE, out_features=MLP_SIZE),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=MLP_SIZE, out_features=EMBED_SIZE),
            nn.Dropout(p=0.1)
        )
    def forward(self, X):
        return self.mlp(self.ln(X))

class TransformerEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.msa = MultiheadSelfAttention()
        self.mlp = MultiLayerPerceptron()
        
    def forward(self, X):
        X = self.msa(X) + X
        X = self.mlp(X) + X
        return X


class LinearClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=EMBED_SIZE)
        self.linear = nn.Linear(in_features=EMBED_SIZE, out_features=3)
        
    def forward(self, X):
        return self.linear(self.ln(X))
    
    
class ViTBase(nn.Module):
    
    def __init__(self) -> None:
        
        super().__init__()
        
        self.patch_embedding = PatchEmbedding()
        self.transformer_encoder = nn.Sequential(*[TransformerEncoder() for _ in range(LAYERS)])
        self.classification_head = LinearClassifier()
   
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        
        X = self.patch_embedding(X)
        X = self.transformer_encoder(X)
        
        return self.classification_head(X[:, 0])

vit_base = ViTBase().to(DEVICE)

# # test if forward pass on single batch works
# vit_base.train()
# logits = vit_base(img_batch.to(DEVICE))
# batch_preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)

# torchinfo.summary(model=vit_base, input_size=(32, 3, 224, 224),
#                   col_names=('input_size', 'output_size', 'num_params', 'trainable'),
#                   )


#----------------------------------
#       TRAINING
#----------------------------------
EPOCHS = 2

optimizer = torch.optim.Adam(params=vit_base.parameters(),
                             lr=0.003,
                             weight_decay=0.3,
                             betas=(0.9, 0.999))

loss_fn = nn.CrossEntropyLoss()

from timeit import default_timer as timer 
start_time = timer()

train(model=vit_base,
      train_dataloader=train_dataloader,
      test_dataloader=test_dataloader,
      optimizer=optimizer,
      loss_fn=loss_fn,
      device=DEVICE)

print(f"Total training time on {DEVICE}: {timer()-start_time:.3f} seconds")

torch.save(obj=vit_base.state_dict(),
           f=Path('models') / 'PizzaSteakSushi_ViT_from_scratch.pth')

#----------------------------------
#       TRANSFER LEARNING
#----------------------------------

pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(DEVICE)
pretrained_vit_transforms = pretrained_vit_weights.transforms()

#-----------------------------
#       DATA
pretrained_vit.heads = nn.Linear(in_features=EMBED_SIZE,
                                 out_features=3).to(DEVICE)
train_dataset = CustomDataset(path=PATH / 'train',
                              transform_input=train_transform)
test_dataset = CustomDataset(path=PATH / 'test',
                             transform_input=test_transform)
train_dataloader = DataLoader(train_dataset,  
                              batch_size=BATCH_SIZE,
                              shuffle=True)
test_dataloader = DataLoader(test_dataset,  
                             batch_size=BATCH_SIZE,
                            shuffle=True)

#-----------------------------
#       MODEL
# torchinfo.summary(pretrained_vit, 
#                   input_size=(32, 3, 224, 224),
#                   col_names=('input_size', 'output_size', 'num_params', 'trainable'))

# print(pretrained_vit)

for param in pretrained_vit.parameters():
    param.requires_grad = False
    
pretrained_vit.heads = nn.Linear(in_features=EMBED_SIZE, 
                                 out_features=3).to(DEVICE)
    
#-----------------------------
#       TRAINING
start_time = timer()

train(model=pretrained_vit,
      train_dataloader=train_dataloader,
      test_dataloader=test_dataloader,
      optimizer=optimizer,
      loss_fn=loss_fn,
      epochs=EPOCHS,
      device=DEVICE)

print(f"Fine-tuning training time on {DEVICE}: {timer()-start_time:.3f} seconds")
