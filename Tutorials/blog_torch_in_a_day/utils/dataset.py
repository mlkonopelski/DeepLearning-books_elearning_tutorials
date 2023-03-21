import os
from pathlib import Path
from typing import Callable

from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as T

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
