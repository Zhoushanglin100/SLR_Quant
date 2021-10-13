import torch
from torchvision import datasets, transforms


class MNISTDataLoader:
    def __init__(self, batch_size, test_batch_size, validate_batch_size, kwargs):
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('/data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.CenterCrop(20),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('/data', train=False, transform=transforms.Compose([
                transforms.CenterCrop(20),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=test_batch_size, shuffle=True, **kwargs)



