from micro_grade import Value
import torch

# -------------------------------------------------------------------------
# PyTorch demo
#
# This example reproduces the same computation graph previously implemented
# with the custom Value class. It illustrates how PyTorch automatically
# builds the computation graph and computes gradients through autograd.
# -------------------------------------------------------------------------

x1 = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
x2 = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
w1 = torch.tensor([-3.0], dtype=torch.float64, requires_grad=True)
w2 = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
b  = torch.tensor([6.8813735870195432], dtype=torch.float64, requires_grad=True)

# Forward pass
n = x1 * w1 + x2 * w2 + b
o = torch.tanh(n)

# Backward pass
o.backward()

print("Output:", o.item())
print()

print("Gradients")
print(f"x1: {x1.grad.item()}")
print(f"x2: {x2.grad.item()}")
print(f"w1: {w1.grad.item()}")
print(f"w2: {w2.grad.item()}")
print(f"b : {b.grad.item()}")


# The same computation can be reproduced with the custom Value class.
# Both implementations should produce identical outputs and gradients.
