# time to train, validate, and test our model!
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"  # this is for mac (Metal Performance Shaders)
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

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

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

# Hyperparameters: let you control model optimization process
learning_rate = 1e-2  # how much to update model after each batch/epoch
batch_size = 64  # number of samples propagated through network before parameters are updated
epochs = 10  # number of times to iterate over the dataset

# after setting hyperparameters we can train our model with an optimization loop
# each iteration of the loop is called an `epoch`
# each epoch consists of two parts:
#   Train Loop - iterate over training data and try converging to optimal parameters
#   Validation/Test Loop - iterate over test data and check if performance is improving

# The Loss Function
# common examples are `nn.MSELoss` for regression, and `nn.NLLLoss` for classification
# `nn.CrossEntropyLoss` combines `nn.LogSoftmax` with `nn.NLLLoss`
loss_fn = nn.CrossEntropyLoss()

# Optimizer
# this is the process of adjusting model parameters after each step
# here we use Stochastic Gradient Descent (SGD) optimization
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training occurs in three steps:
# - call `optimizer.zero_grad()` to reset gradients of model parameters
# - backpropagate the prediction loss with a call to `loss.backward()`
# - once we have the gradients call `optimizer.step()` to adjust parameters by the gradients

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set model to training mode
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluate the model with torch.no_grad()
    # ensures no gradients are computed, preserving parameters and avoiding unnecessary calculations
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

print("Done!")
