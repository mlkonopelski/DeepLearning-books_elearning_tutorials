from torchvision import datasets
from torchvision import transforms


if __name__ == '__main__':
    data_path = '.data/'
    train_dataloader = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    test_dataloader = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    #EDA
    X, y = next(train_dataloader)
    
    print(type(X), type(y))
    print(X.shape, y.shape)
    
    