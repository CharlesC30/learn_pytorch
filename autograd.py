# backpropagation is used for adjusting model parameters during training
# `torch.autograd` is a built-in differentiation engine for computing gradients

# consider the simplest one-layer network 
# x = input, w, b = parameters
import torch

x = torch.ones(5)  # input
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# `requires_grad=True` allows us to calculate the gradient of the loss fucntion
# w.r.t. those tensors

# `z` and `loss` here are objects of the `Function` class 
# this class knows how to compute in the forward direction and calculate the derivate
# in the backward direction
# `grad_fn` property stores a reference for the back propagation\
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# computing the gradients w.r.t. `w` and `b`
loss.backward()
print(w.grad)
print(b.grad)

# if we want to stop tracking gradients (e.g. done training and just want to apply model)
# then we can wrap code in `torch.no_grad()` block
z = torch.matmul(x, w) + b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)

# another way is to use the `detach()` method on our tensor
z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad)



