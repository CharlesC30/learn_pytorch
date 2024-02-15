# Sometimes data is not loaded in the form we want to train with
# transforms manipulate data so it can be used for training
# TorchVision Datasets have `transform` and `target_transform` parameters
# these operate on features and labels, respectively

# `torchvision.transforms` has many common transforms for working with images

# FashionMNIST features are in PIL format and labels are integers
# we need to convert these to tensors and "one-hot encoded tensors"

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),  # converts to FloatTensor and scales values to [0., 1.]
    # this lambda functions makes it so the labels are always of size 10 
    # with a 1 in the corresponding label spot and 0's elsewhere
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)


