import torch

x_train, y_train, x_valid, y_valid = ..., ..., ..., ...

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)

