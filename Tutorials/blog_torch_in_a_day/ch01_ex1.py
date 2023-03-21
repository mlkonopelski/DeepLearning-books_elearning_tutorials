import torch
from torch import nn
from pathlib import Path
from datetime import datetime

DEVICE = 'mps'
EPOCHS = 1000

weight = 0.3
bias = 0.9

X = torch.arange(start=0, end=10000, step=0.01).unsqueeze(dim=1)
y = weight * X + bias

split_idx = int(len(X) * 0.8)

X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

# model
class LinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Linear(1, 1)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)
    
model = LinearModel().to(DEVICE)
print(model)

# loss and optim
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-2)

# training
start = datetime.now()
for epoch in range(EPOCHS):
    X_train = X_train.to(DEVICE)
    y_train = y_train.to(DEVICE)
    X_test = X_test.to(DEVICE)
    # train
    model.train()
    y_preds = model(X_train)
    loss = loss_fn(y_train, y_preds)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # eval
    model.eval()
    with torch.inference_mode():
        y_preds = model(X_test)
        test_loss = loss_fn(y_test, y_preds.to('cpu'))
        
    if epoch % 100 == 0:
        print(f'EPOCH: {epoch} | train loss: {loss} | test loss: {test_loss}')
print(f'Training finished in {datetime.now() - start}')

# model name
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "01_pytorch_workflow_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

torch.save(obj=model.state_dict(),
           f=MODEL_SAVE_PATH)
