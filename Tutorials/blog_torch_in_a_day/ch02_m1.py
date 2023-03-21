from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

n_samples = 10000

DEVICE = 'mps' if torch.has_mps else 'cpu'
print(DEVICE)

X, y = make_circles(n_samples, random_state=12345, noise=0.03)

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=12345, shuffle=True)

print(f'{X_train.size()}|{y_train.size()}\n{X_test.size()}|{y_test.size()}')

X_train = X_train.to(DEVICE)
y_train = y_train.to(DEVICE)
X_test = X_test.to(DEVICE)
y_test = y_test.to(DEVICE)


class LinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(2, 10)
        self.l12 = nn.Linear(10, 10)
        self.l2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.l1(input))
        y = self.relu(self.l12(y))
        y = self.l2(y)
        y = self.sigmoid(y)
        return y

model = LinearModel()
model.to(DEVICE)

print(model)

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-1)

def accuracy(y_true, y_prob):
    y_pred = torch.round(y_prob)
    correct = torch.eq(y_true, y_pred).sum().item()
    all = len(y_true)
    return ((correct / all) * 100)


EPOCHS = 1000
for epoch in range(EPOCHS):
    # train
    model.train()
    y_preds = model(X_train).squeeze()
    loss = loss_fn(y_preds, y_train)
    acc_train = accuracy(y_train, y_preds)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # eval
    if epoch % 100 == 0:
        model.eval()
        with torch.torch.inference_mode():
            y_preds = model(X_test)
            acc_test = accuracy(y_test, y_preds.squeeze())
            print(f'EPOCH : {epoch} | train: {acc_train:.5f} | test: {acc_test:.5f}')
            
