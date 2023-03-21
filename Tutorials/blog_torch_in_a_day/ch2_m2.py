import torch
from torch import nn
from torch import Tensor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from torchmetrics import Accuracy

DEVICE = 'cpu' #mps
N_SAMPLES = 10000
N_CLASS = 20
N_FEAT = 10
EPOCHS = 1000

X, y = make_blobs(n_samples=N_SAMPLES,
                  n_features=N_FEAT,
                  centers=N_CLASS,
                  cluster_std=1.5,
                  random_state=12345)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345, shuffle=True)

X_train = torch.from_numpy(X_train).type(torch.float).to(DEVICE)
X_test = torch.from_numpy(X_test).type(torch.float).to(DEVICE)
y_train = torch.from_numpy(y_train).type(torch.LongTensor).to(DEVICE)
y_test = torch.from_numpy(y_test).type(torch.LongTensor).to(DEVICE)

print(f'sizes: ({X_train.size()}, {y_train.size()}), ({X_test.size()},{y_test.size()})')


class MultiLabelLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(N_FEAT, 10)
        self.l2 = nn.Linear(10, 10)
        self.l3 = nn.Linear(10, N_CLASS)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        
    def forward(self, input: Tensor) -> Tensor:
        y = self.l1(input)
        y = self.relu(y)
        y = self.relu(self.l2(y))
        y = self.softmax(self.l3(y))
        return y
    
model = MultiLabelLinear().to(DEVICE)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-1)
torchmetrics_accuracy = Accuracy(task='multiclass', num_classes=N_CLASS).to(DEVICE)

for epoch in range(EPOCHS):
    model.train()
    y_preds_prob = model(X_train)
    loss = loss_fn(y_preds_prob, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        model.eval()
        with torch.inference_mode():
            y_preds_prob = model(X_test)
            loss = loss_fn(y_preds_prob, y_test)
            y_preds = torch.argmax(y_preds_prob, dim=1)
            acc = torchmetrics_accuracy(y_preds, y_test)
            print(f'EPOCH: {epoch} loss: {loss:.5f} acc: {acc:.5f}')

