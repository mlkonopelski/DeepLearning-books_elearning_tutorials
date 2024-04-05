import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from Books.deep_learning_with_pytorch._classification_model import ResNet
from Books.deep_learning_with_pytorch._classification_training import train_step, test_step

##########################
#       CONST
EPOCHS = 1
DEVICE = 'mps' 

##########################
#       DATA
data_path = '.data/'
train_dataloader = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transforms.ToTensor())
test_dataloader = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transforms.ToTensor())

##########################
#       MODEL
resnet = ResNet(blocks=1, channels=10)
resnet.to(DEVICE)

##########################
#       TRAINING

adam_opt = torch.optim.Adam(params=resnet.parameters(), lr=1e-3)
bce_ll = nn.BCEWithLogitsLoss()

for epoch in range(EPOCHS):
    train_loss, train_acc = train_step(model=resnet, 
                                       epochs=EPOCHS, 
                                       optimizer=adam_opt, 
                                       loss_fn=bce_ll, 
                                       data_loader=train_dataloader, 
                                       device=DEVICE)
    test_loss, test_acc = test_step(model=resnet, 
                                    loss_fn=bce_ll, 
                                    data_loader=test_dataloader, 
                                    device=DEVICE)
    print(f'epochs: {epoch}\t'
          f'train_loss:{train_loss}\ttrain_acc:{train_acc}'
          f'test_loss:{test_loss}\ttest_acc:{test_acc}')
