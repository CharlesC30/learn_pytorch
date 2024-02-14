import torch
import numpy as np

# tensors are much like numpy arrays, except they can also run on GPU's and hardware accelerators
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

arr = np.array(data)
x_np = torch.from_numpy(arr)

# creating tensors from exising ones
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# "shape" determines tensor dimensions
shape = (2, 3, )
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# attributes can tell us tensor shape, dtype, and where the are stored
tensor = torch.rand(3, 4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}\n")

# tensor operations
# see comprehensize documentation here: https://pytorch.org/docs/stable/torch.html

# operations can be performed on GPU after moving tensors there with `.to` method
tensor = torch.ones(4, 4)
if torch.cuda.is_available():  # always check if available first
    print("Cuda is available")
    tensor = tensor.to("cuda")  # keep in mind copying to GPU can be a slow/expensive operation
tensor[:, 1] = 0
print(tensor)

# join tensors with `.cat`
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(f"Concatenated tensor:\n {t1}\n")

# matrix multiplication 
# these all do the same thing (y1 = y2 = y3)
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

# element-wise multiplication
# these all do the same thing (z1 = z2 = z3)
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# single-element tensors
# tensors with only one element can be converted to Python numerical using `item()`
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# in-place operations
# denoted with `_` suffix
x = torch.rand_like(tensor)
x.copy_(tensor)
x.add_(5)
print(x)

# tensors stored on the CPU can share underlying memory with numpy arrays
# changing one will then change the other!
t = torch.ones(5)
n = t.numpy()
print("\nbefore operation:")
print(f"t: {t}")
print(f"n: {n}\n")

t.add_(1)
print("\nafter operation:")
print(f"t: {t}")
print(f"n: {n}\n")

# same idea but modifying the numpy array
n = np.ones(5)
t = torch.from_numpy(n)
print("\nbefore operation:")
print(f"t: {t}")
print(f"n: {n}\n")

np.add(n, 1, out=n)
print("\nafter operation:")
print(f"t: {t}")
print(f"n: {n}\n")