# All TorchVision datasets have two parameters -transform to modify the features and 
# target_transform to modify the labels - that accept callables containing the transformation logic. 
# The torchvision.transforms module offers several commonly-used transforms out of the box.

import torch 
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))


ds = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=target_transform
)


print(ds[0])

