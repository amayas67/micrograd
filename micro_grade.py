import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph


class Value:
    """Scalar value with autograd support.

    Each Value node tracks:
    - data: the scalar value
    - grad: gradient of the output with respect to this node
    - _prev: parent nodes in the computation graph
    - _op: operation that produced this node
    - _backward: function to compute local gradients
    - label: for visualization
    """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0  # gradient, initialized to zero
        self._prev = set(_children)  # parent nodes in computation graph
        self._backward = lambda: None  # default: no backward pass (leaf node)
        self._op = _op  # operation that produced this node
        self.label = label  # for visualization

    def __add__(self, other):
        """Addition: out = self + other"""
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(out)/dself = 1, d(out)/dother = 1
            # chain rule: dL/dself = dL/dout * d(out)/dself = out.grad * 1
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        """Multiplication: out = self * other"""
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # d(out)/dself = other.data, d(out)/dother = self.data
            # chain rule: dL/dself = dL/dout * other.data
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        """Hyperbolic tangent: out = tanh(self)"""
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # d(tanh(x))/dx = 1 - tanh(x)^2 = 1 - t^2
            # chain rule: dL/dself = dL/dout * (1 - t^2)
            self.grad += out.grad * (1 - t**2)

        out._backward = _backward
        return out

    def __repr__(self):
        return f"Value(data={self.data})"

    def topological_order_reversed(self):

        """
        Returns a topological ordering of the computation graph starting from this node.

        CRITICAL BUG NOTE (Fixed):
        - Old implementation appended nodes to 'visited' BEFORE exploring their parents (Pre-order).
          In graphs where a variable is reused multiple times (e.g., y = a * a), this caused
          _backward() to execute prematurely on shared nodes before they accumulated gradients
          from all downstream branches.
        - Fix implementation uses Depth-First Search (DFS) with Post-order recording. A node
          is added to the list ONLY after all its dependencies are fully explored. Reversing
          this list guarantees a valid backpropagation sequence (Output -> Intermediates -> Inputs).
        """
        visited = []

        def topological_order_r(node):
            if node not in visited:
                # 1. Deeply explore all parent nodes (dependencies) first
                for child in node._prev:
                    topological_order_r(child)
                # 2. Business Logic: Append the node ONLY after all its parents are visited (Post-order)
                visited.append(node)

        # Trigger the DFS traversal starting from the output node (self)
        topological_order_r(self)

        # 3. Reverse the post-order list to flow from output back to inputs for backpropagation
        return list(reversed(visited))

    def cal_backward(self):
        list_ = self.topological_order_reversed()
        for node in list_:
            node.grad = 0.0
        self.grad = 1.0
        for node in list_:
            node._backward()



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

        # Draw operation node (circle) if this node is result of an operation
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    # Draw edges from children to operation nodes
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot




# =============================================================================
# TEST: Complex computation graph
# =============================================================================

if __name__ == "__main__":
    # Build complex graph: i = (a*b + (c+f)) * h
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    f = Value(-2.0, label='f')
    h = Value(3.0, label='h')

    e = a * b; e.label = 'e'      # e = -6
    d = c + f; d.label = 'd'      # d = 8
    g = e + d; g.label = 'g'      # g = 2
    i = g * h; i.label = 'i'      # i = 6


    i.cal_backward()
    print(i.topological_order_reversed())
    # Print results
    print("\n=== GRADIENTS ===")
    print(f"i.grad = {i.grad}")   # 1.0
    print(f"g.grad = {g.grad}")   # 3.0
    print(f"h.grad = {h.grad}")   # 2.0
    print(f"e.grad = {e.grad}")   # 3.0
    print(f"d.grad = {d.grad}")   # 3.0
    print(f"c.grad = {c.grad}")   # 3.0
    print(f"f.grad = {f.grad}")   # 3.0
    print(f"a.grad = {a.grad}")   # -9.0
    print(f"b.grad = {b.grad}")   # 6.0

    # # Draw and save graph
    # dot = draw_dot(i)
    # dot.render('computation_graph_complex', view=False)
    # print("\nGraph saved to computation_graph_complex.svg")
