import os
from pathlib import Path
from typing import Callable, Tuple, Dict, List 

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

#print(torch.__version__)

WORKERS = os.cpu_count()
# WORKERS = 1
DATAPATH = Path('data') / 'pizza_steak_sushi'
BATCH_SIZE = 32
DEVICE = 'mps'
    
train_transform = T.Compose([
    T.Resize(size=(224, 224)),
    T.TrivialAugmentWide(num_magnitude_bins=31),
    T.ToTensor()
])

test_transform = T.Compose([
    T.Resize(size=(224, 224)),
    T.ToTensor()
])

# # EASY WAY OF BUILDING CUSTOM DATASETS
# train_dataset = ImageFolder(root=DATAPATH / 'train',
#                             transform=train_transform,
#                             target_transform=None)
# test_dataset = ImageFolder(root=DATAPATH / 'test',
#                            transform=test_transform)

# print(f"Train data:\n{train_dataset}\nTest data:\n{test_dataset}")
# print(train_dataset.class_to_idx)

# img, label = train_dataset[0][0], train_dataset[0][1]
# print(f'Image size: {img.size()}')
# print(f'Label: {train_dataset.classes[label]}')
# plt.imshow(img.permute(1, 2, 0))  # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])


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

# print(f"Train data:\n{train_dataset}\nTest data:\n{test_dataset}")
# print(train_dataset.class_to_idx)

# img, label = train_dataset[0][0], train_dataset[0][1]
# print(f'Image size: {img.size()}')
# print(f'Label: {train_dataset.classes[label]}')
# plt.imshow(img.permute(1, 2, 0))  # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=WORKERS
                              )

test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=WORKERS)


# # ADHOC TEST
# file_count = 0
# for X, y in train_dataloader:
#     file_count += len(X)

# # Check if augmentation increased number of samples per EPOCH. Answer: NO
# print(sum([len(os.listdir(DATAPATH / 'train' / folder)) for folder in os.listdir(DATAPATH / 'train')]))
# print(file_count)


class TinyVGG(nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=int(hidden_units * ((224 / 4) ** 2)),
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion


def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer):
    # Put model in train mode
    model.train()
    
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target DEVICE
        X, y = X.to(DEVICE), y.to(DEVICE)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc



def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module):
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(DEVICE), y.to(DEVICE)
    
            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


from tqdm.auto import tqdm

# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn)
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return results

if __name__ == '__main__':
    # Set random seeds
    torch.manual_seed(42) 
    torch.cuda.manual_seed(42)

    # Set number of epochs
    NUM_EPOCHS = 5

    # Recreate an instance of TinyVGG
    model_0 = TinyVGG(input_shape=3, # number of color channels (3 for RGB) 
                    hidden_units=10, 
                    output_shape=len(train_dataset.classes)).to(DEVICE)

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

    # Start the timer
    from timeit import default_timer as timer 
    start_time = timer()

    # Train model_0 
    model_0_results = train(model=model_0, 
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            optimizer=optimizer,
                            loss_fn=loss_fn, 
                            epochs=NUM_EPOCHS)

    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")
