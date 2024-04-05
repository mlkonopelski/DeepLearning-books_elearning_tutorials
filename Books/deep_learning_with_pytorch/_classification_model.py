import torch.nn as nn
import torch
import torch.functional as F


##############################
#       RES NET

class ResBlock(nn.Module):
    def __init__(self, channels: int = 10, kernel_size: int = 2) -> None:
        super(ResBlock).__init__()
        self.conv = nn.Conv2d(in_channels=channels, 
                              out_channels=channels,
                              kernel_size=kernel_size,
                              paddng=1,
                              bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=channels)
        torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.batch_norm.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        return out + x

class ResNet(nn.Module):
    def __init__(self, blocks: int, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(in_channels=3,
                              out_channels=channels,
                              kernel_size=3,
                              padding=1)
        self.resblocks = nn.Sequential(*[ResBlock(channels=channels) for _ in range(blocks)])
        self.fc1 = nn.Linear(in_features=8*8*channels, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=10)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.max_pool2d(torch.relu(self.conv(x)), 2)
        out = self.resblocks(out)
        out = out.view(-1, 8 * 8 * self.channels)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

