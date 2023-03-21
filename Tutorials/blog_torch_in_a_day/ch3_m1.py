import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple
from timeit import default_timer as timer
from tqdm.auto import tqdm


BATCH_SIZE = 32
DEVICE = 'cpu' # 'mps'
EPOCHS=10

data_path = Path('data')
data_path.mkdir(exist_ok=True)

def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

train_dataset = datasets.FashionMNIST(root='data',
                                      train=True,
                                      transform=ToTensor(),
                                      target_transform=None,
                                      download=True)

test_dataset = datasets.FashionMNIST(root='data',
                                     train=False,
                                     transform=ToTensor(),
                                     download=True)

class_names = train_dataset.classes

# img, label = test_dataset[0]
# print(img.size(), img.squeeze().size())
# plt.imshow(img.squeeze(), cmap='gray')

train_dataloader = DataLoader(train_dataset, 
                              batch_size=BATCH_SIZE,
                              shuffle=True
                              )
test_dataloader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=True
                             )


class LinearModel(nn.Module):
    def __init__(self, img_size: Tuple[int, int], classes_num: int,  hidden_units: int = 100) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=img_size[0] * img_size[1],
                                out_features=100)
        self.linear2 = nn.Linear(in_features=100, out_features=hidden_units)
        self.linear3 = nn.Linear(in_features=hidden_units,
                                 out_features=classes_num)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.flatten(X)
        X = self.relu(self.linear1(X))
        X = self.relu(self.linear2(X))
        y = self.softmax(self.linear3(X))
        return y
    
linear_model = LinearModel(img_size=(28, 28),
                           classes_num=10,
                           hidden_units=len(class_names)) \
                .to(DEVICE)

accuracy = Accuracy(task='multiclass', num_classes=len(class_names)).to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=linear_model.parameters(), 
                            lr =1e-1)

start = timer()
for epoch in range(EPOCHS):
    print(f'EPOCH: {epoch}')
    
    # TRAINING
    linear_model.train()
    train_loss, train_acc = 0, 0
    train_batches = len(train_dataloader)
    for X_train, y_train in train_dataloader:
        X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
        y_train_pred_prob = linear_model(X_train)
        y_train_pred = torch.argmax(y_train_pred_prob, dim=1)
        loss = loss_fn(y_train_pred_prob, y_train)
        acc = accuracy(y_train_pred, y_train.type(torch.LongTensor))
        train_loss += loss
        train_acc += acc
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss = train_loss / train_batches
    train_acc = train_acc / train_batches
    print(f'\tTRAIN loss: {train_loss:.5f} accuracy: {train_acc:.2f}')
    
    # EVALUATION
    linear_model.eval()
    test_loss, test_acc = 0, 0
    test_batches = len(test_dataloader)
    for X_test, y_test in test_dataloader:
        X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)
        with torch.inference_mode():
            y_test_pred_proba = linear_model(X_test)
            y_test_pred = torch.argmax(y_test_pred_proba, dim=1)
            test_loss += loss_fn(y_test_pred_proba, y_test)
            test_acc += accuracy(y_test_pred, y_test.type(torch.LongTensor))
    test_loss = test_loss / test_batches
    test_acc = test_acc / test_batches 
    print(f'\tTEST loss: {test_loss:.5f} accuracy: {test_acc:.2f}')

print_train_time(start=start, end=timer(), device=DEVICE)
