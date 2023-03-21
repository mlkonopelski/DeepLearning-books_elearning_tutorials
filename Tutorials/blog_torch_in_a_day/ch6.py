import os
from pathlib import Path
from typing import Callable, Tuple, Dict, List 

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

from torchinfo import summary

#print(torch.__version__)

#WORKERS = os.cpu_count()
WORKERS = 1
DATAPATH = Path('data') / 'pizza_steak_sushi'
BATCH_SIZE = 32
DEVICE = 'mps'
    
train_transform = T.Compose([
    T.Resize(size=(224, 224)),
    T.TrivialAugmentWide(num_magnitude_bins=31),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
                std=[0.229, 0.224, 0.225])
])

test_transform = T.Compose([
    T.Resize(size=(224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
                std=[0.229, 0.224, 0.225])
])

# # ADVANCE WAY OF BUILDING CUSTOM DATASETS
def find_labels(path):
    labels = sorted(os.listdir(path))
    labels_dict = {}
    for i, label in enumerate(labels):
        labels_dict[label] = i
    return labels, labels_dict

class CustomDataset(Dataset):
    
    __name__ = 'CustomDataset'
    
    def __init__(self, path, transform_input: Callable = None) -> None:
        super().__init__()
        self.path = path
        self.img_paths = list(path.glob('*/*.jpg'))
        self.classes, self.class_to_idx = find_labels(path)
        self.transorm_input = transform_input

    def __len__(self):
        return sum([len(os.listdir(self.path / folder)) for folder in os.listdir(self.path)])
    
    def __str__(self) -> str:
        txt = f'{self.__name__}:\n'
        txt += f'Path: {self.path}\n'
        txt += f'DataPoints: {self.__len__()}\n'
        txt += f'Loading: images read in __getitem__\n'
        txt += f'Transformation:\n{self.transorm_input}\n'
        return txt
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path)
        label = img_path.parent.name
        label_idx = self.class_to_idx[label]
        
        if self.transorm_input:
            img = self.transorm_input(img)
            
        return img, label_idx

train_dataset = CustomDataset(path=DATAPATH / 'train',
                              transform_input=train_transform)
test_dataset = CustomDataset(path=DATAPATH / 'test',
                              transform_input=test_transform)

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              #num_workers=WORKERS
                              )

test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             #num_workers=WORKERS
                             )


weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=weights)

# summary(model,
#         input_size=(32, 3, 224, 224),
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"])

for param in model.features.parameters():
    param.requires_grad = False
    
# summary(model,
#         input_size=(32, 3, 224, 224),
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"])

model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280,
              out_features=len(train_dataset.classes),
              bias=True)
    )

model = model.to(DEVICE)

# summary(model,
#         input_size=(32, 3, 224, 224),
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"])

from utils.training import train

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)


if __name__ == '__main__':

    train(model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=5)
