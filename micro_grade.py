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

    def __rsub__(self, other):
        return other + (-self)
    def __add__(self, other):
        """Addition: out = self + other"""
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(out)/dself = 1, d(out)/dother = 1
            # chain rule: dL/dself = dL/dout * d(out)/dself = out.grad * 1
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return  self + other

    def __mul__(self, other):
        """Multiplication: out = self * other"""
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')


        def _backward():
            # d(out)/dself = other.data, d(out)/dother = self.data
            # chain rule: dL/dself = dL/dout * other.data
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self ,other):
        return self + (-other)



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

    def exp(self):
        x= self.data
        out = Value(math.exp(x),(self,),'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def __truediv__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        return self*(other**(-1))

    def __rtruediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other * (self ** -1)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Exponent must be an int or float"
        out = Value(self.data ** other, (self,), f"**{other}")
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
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

    # -------------------------------------------------------------------------
    # Validation test
    #
    # tanh(x) can be implemented directly or using the identity:
    #
    #     tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    #
    # Both implementations should produce identical gradients.
    # Tiny numerical differences (~1e-15) are expected due to IEEE 754
    # floating-point rounding.
    # -------------------------------------------------------------------------

    # Build the computation graph
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    f = Value(-2.0, label='f')
    h = Value(3.0, label='h')

    e = a * b; e.label = 'e'
    d = c + f; d.label = 'd'
    g = e + d; g.label = 'g'
    i = g * h; i.label = 'i'

    variables = {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "e": e,
        "f": f,
        "g": g,
        "h": h,
        "i": i,
    }

    # ---------------- First implementation ----------------
    k1 = i.tanh()
    k1.cal_backward()

    grads_tanh = {name: v.grad for name, v in variables.items()}

    # ---------------- Second implementation ----------------
    k2 = ((2 * i).exp() - 1) / ((2 * i).exp() + 1)
    k2.cal_backward()

    grads_exp = {name: v.grad for name, v in variables.items()}

    # ---------------- Comparison ----------------
    print(f"{'Var':<3} {'tanh':>20} {'exp identity':>20} {'|Δ|':>15}")
    print("-" * 65)

    for name in variables:
        diff = abs(grads_tanh[name] - grads_exp[name])
        print(
            f"{name:<3} "
            f"{grads_tanh[name]:>20.16e} "
            f"{grads_exp[name]:>20.16e} "
            f"{diff:>15.3e}"
        )

# Draw and save the computation graph
    dot = draw_dot(k1) # or k2
    dot.render("computation_graph_complex", view=False)
    print("\nGraph saved to computation_graph_complex.svg")
