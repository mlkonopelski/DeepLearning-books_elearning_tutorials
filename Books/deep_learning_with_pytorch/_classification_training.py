import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

def accuracy(y_true, y_pred):
    return y_true == y_pred

def train_step(model: nn.Module, epochs: int, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, data_loader: DataLoader, device: str):
    # set model in train mode
    model.train()
    
    train_loss = 0
    train_acc = 0
    
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        # forward pass
        y_logit = model(X)
        # calculate gradients
        loss = loss_fn(y_logit, y)
        train_loss += loss
        train_acc += (y_logit.argmax(1) == y).sum().item()
        # zero gradients from previous run
        optimizer.zero_grad()
        # calculate gradients
        loss_fn.backward()
        # update weights
        optimizer.step()
    
    train_loss = train_loss / len(data_loader)
    train_acc = train_acc / len(data_loader)
    
    return train_loss, train_acc


def test_step(model: nn.Module, loss_fn: nn.Module, data_loader: DataLoader, device: str):
    # set model in eval mode
    model.eval()
    
    test_loss = 0
    test_acc = 0
    
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            # forward pass
            y_logit = model(X)
        # calculate loss
        loss = loss_fn(y_logit, y)
        test_loss += loss
        test_acc += (y_logit.argmax(1) == y).sum().item()
    
    
    train_loss = train_loss / len(data_loader)
    train_acc = train_acc / len(data_loader)
    
    return train_loss, train_acc