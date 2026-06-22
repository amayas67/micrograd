import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph


class Value:
    """Scalar value with autograd support."""

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0  # gradient, initialized to zero
        self._prev = set(_children)  # parent nodes in computation graph
        self._backward = lambda:None
        self._op = _op  # operation that produced this node
        self.label = label  # for visualization

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


# =============================================================================
# VISUALIZATION
# =============================================================================

def trace(root):
    """Build set of all nodes and edges in the computation graph."""
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    """Draw computation graph using graphviz."""
    nodes, edges = trace(root)
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

    for n in nodes:
        uid = str(id(n))
        # Node label with data and grad
        label = f"data {n.data:.4f} | grad {n.grad:.4f}"
        if n.label:
            label = f"{n.label} | {label}"

        dot.node(name=uid, label=label, shape='record')

        # Operation node
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


# =============================================================================
# TEST
# =============================================================================

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b; e.label = 'e'
d = e + c; d.label = 'd'

d.grad = 1
d._backward()
e._backward()

# Draw and save graph
dot = draw_dot(d)
dot.render('computation_graph', view=False)
print("Graph saved to computation_graph.svg")

print(f"d = {d}")
print(f"d._prev = {d._prev}")
print(f"d._op = '{d._op}'")
print(f"d.tanh  {d.tanh()} ")
print(f"d.grad_glob  {d.grad} ")
