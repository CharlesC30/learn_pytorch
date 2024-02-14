# ideally we want to separate data processing code from model training
# two data primitives: `torch.utils.data.DataLoader` and `torch.utils.data.Dataset`
# Dataset stores samples and labels, DataLoader provider iterable wrapper for easy access

# PyTorch provides sample datasets that subclass `torch.utils.data.Dataset`

# example loading FashionMNIST Dataset

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="data",  # path where data is stored
    train=True,  # use as training dataset
    download=True,  # download the data if not in root
    transform=ToTensor(),  # specify feature and label transformations
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# we can iterate over datasets just like a list
# visualize some of the data with matplotlib
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()


# Creating custom datasets
# custom datasets must implement `__init__`, `__len__`, and `__getitem__`

# in this implementation the FashionMNIST images are stored in `img_dir`
# and their labels are in `annotations_file`
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):

    # tell class where the images are stored, the labels are stored, and the transform functions
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    # return number of samples in the dataset
    def __len__(self):
        return len(self.img_labels)

    # loads and returns a sample based on the input index
    # first find the image location on disk, convert it to a tensor, and retrives corresponding label
    # sample and label are finally return in as a pair in a tuple
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
# When training we want to process data in "minibatches" to reduce overfitting
# we also want to use Python's multiprocessing to speed up data retrieval
# `DataLoader` lets us abstract away this complexity
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# iterating over the DataLoader
# each iteration returns 64 (batch size) train_features and train_labels
# shuffle=True tells the DataLoader to shuffle the data once we iterate over everything
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

