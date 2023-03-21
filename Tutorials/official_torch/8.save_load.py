import torch
import torchvision.models as models
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.seq_layers = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.seq_layers(x)
        return logits
    
linear_model = NeuralNetwork()
torch.save(linear_model.state_dict(), 'models/linear_model_weights.pth')


loaded_model = NeuralNetwork()
loaded_model.load_state_dict(torch.load('models/linear_model_weights.pth'))
loaded_model.eval() # be sure to call model.eval() method before inferencing to set the dropout and batch normalization layers to evaluation mode. Failing to do this will yield inconsistent inference results.


torch.save(linear_model, 'models/linear_model.pth')
linear_model = torch.load('models/linear_model.pth')
