import math
import numpy as np
import matplotlib.pyplot as plt


class Value:
    """Scalar value with autograd support."""

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0  # gradient, initialized to zero
        self._prev = set(_children)  # parent nodes in computation graph
        self._backward = lambda:None
        self._op = _op  # operation that produced this node

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad = out.grad
            other.grad = out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad
        out._backward = _backward
        return out

    def __repr__(self):
        return f"Value(data={self.data})"

    def tanh(self):
        x = self.data
        out= Value((math.exp(2*x)-1) / (math.exp(2*x)+1),(self,),'tanh')
        def _backward():
            self.grad = out.grad* (1-out.data**2)
        out._backward = _backward
        return out


# Test: build computation graph
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a * b + c

print(f"d = {d}")
print(f"d._prev = {d._prev}")
print(f"d._op = '{d._op}'")
print(f"d.tanh  {d.tanh()} ")
