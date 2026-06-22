import math
import numpy as np
import matplotlib.pyplot as plt


class Value:
    """Scalar value with autograd support."""

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0  # gradient, initialized to zero
        self._prev = set(_children)  # parent nodes in computation graph
        self._op = _op  # operation that produced this node

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out

    def __repr__(self):
        return f"Value(data={self.data})"


# Test: build computation graph
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a * b + c

print(f"d = {d}")
print(f"d._prev = {d._prev}")
print(f"d._op = '{d._op}'")
