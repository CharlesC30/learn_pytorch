# `torch.nn` namespace provides building blocks for neural networks
# all NN modules subclass `nn.Module`

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# we want to train on the GPU if possible
# so let's check if it's available and use it

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"  # this is for mac (Metal Performance Shaders)
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# define our NN by subclassing `nn.Module`
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    # every `nn.Module` subclass implements operations on input data in `forward`
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# initialize the model
model = NeuralNetwork().to(device)
print(model)
# to use the model pass it input data, do not call `model.forward()` directly!

X = torch.rand(1, 28, 28, device=device)
# output of the model is a 2d tensor
# dim=0 corresponds to each output of 10 raw predicted values
# dim=1 corresponds to individual values of each output
logits = model(X)
# print(logits)
pred_probab = nn.Softmax(dim=1)(logits)  # convert raw scores into probabilities
# print(pred_probab)
y_pred = pred_probab.argmax(1)  # find label with highest probability
print(f"Predicted class: {y_pred}")


# Breaking down model layers

# `nn.Flatten` layer converts 2D 28x28 images into contiguous array of 784 pixel values
input_image = torch.rand(3, 28, 28)  # three random images
flatten = nn.Flatten()
flat_image = flatten(input_image)  # flatten the images, by default minibatch dimension (dim=0) is maintained
print(f"Original size: {input_image.size()}, Flattened size: {flat_image.size()}")

# `nn.Linear` applies linear transform using stored weights and biases
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(f"After first linear transform, size = {hidden1.size()}\n")

# `nn.ReLU` introduces non-linearity to the model 
# non-linearity lets models create complex mappings and learn wide variety of phenomena
# here ReLU is used but other non-linear activations exist
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# `nn.Sequential` is an ordered container of modules
# data is passed in the same order as defined\
# can be used to put together a quick network
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

# Finally `nn.Softmax` scales logit outputs from model into range [0, 1]
# this represents the predicted probabilities of each class 
# `dim` parameter indicates along which dimension values should sum to 1
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
print(f"Logits: {logits}")
print(f"Predicted probabilities: {pred_probab}\n\n")

# Model Parameters
# many layers inside the network are 'parameterized'
# i.e., they have associated weights and biases that are learned during training
# subclassing `nn.Module` makes all parameters accessible via `parameters()` or `name_parameters()` methods
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")