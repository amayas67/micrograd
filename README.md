# micrograd — Scalar Autograd Engine from Scratch

A full reimplementation of a scalar-valued automatic differentiation engine,
built independently before consulting reference solutions.

---

## What this is

A from-scratch implementation of backpropagation over a dynamically built
computation graph — the same mechanism that powers PyTorch, JAX, and every
modern deep learning framework.

The engine is scalar: every value in the graph is a single float. This
constraint strips away all the complexity of tensor broadcasting and exposes
the pure mathematical structure of gradient computation.

---

## What is implemented

### Core engine — `micro_grade.py`

The `Value` class wraps a scalar and tracks everything needed for autograd:

| Component | Description |
|---|---|
| `data` | The scalar value |
| `grad` | Gradient of the output w.r.t. this node |
| `_prev` | Parent nodes in the computation graph |
| `_backward` | Closure that computes local gradients |

**Supported operations:**

`__add__`, `__radd__`, `__mul__`, `__rmul__`, `__pow__`, `__neg__`,
`__sub__`, `__rsub__`, `__truediv__`, `__rtruediv__`, `tanh`, `exp`

Every operation:
1. Computes the forward value
2. Registers a `_backward` closure implementing the local chain rule
3. Tracks parent nodes for graph traversal

**Backward pass:**

```python
loss.cal_backward()
```

Internally:
- Builds a **post-order topological sort** via DFS (children before parents)
- Resets all gradients to `0.0`
- Sets `self.grad = 1.0` (seed gradient)
- Calls `_backward()` on each node in reverse topological order

**Why `+=` and not `=` in every `_backward`:**

When a node appears in multiple paths of the graph (e.g. `y = a * a`),
its gradient must accumulate contributions from every path.
This is the multivariate chain rule:

```
dL/da = dL/de1 * de1/da  +  dL/de2 * de2/da  +  ...
```

The `+=` is not a convention — it is a mathematical necessity.

---

### Neural network — `MLP_object.py`

Built on top of the autograd engine:

```
Neuron   — tanh(w·x + b), exposes parameters()
Layer    — collection of independent neurons
MLP      — sequence of layers, forward propagation
```

`parameters()` is implemented at every level (Neuron → Layer → MLP),
allowing a single call to retrieve all trainable weights and biases.

---

### Training loop — `train_demo.py`

Full gradient descent loop:

```python
for epoch in range(1000):
    ypred, loss = recalculate(n, xs, ys)   # forward pass + MSE loss
    loss.cal_backward()                     # backward pass
    for p in n.parameters():
        p.data -= learning_rate * p.grad   # gradient descent step
```

Note: `cal_backward()` resets all gradients before each backward pass.
In PyTorch this must be done explicitly with `optimizer.zero_grad()`.

---

### Validation against PyTorch — `pytorch_demo.py`

Gradients computed by the custom engine are verified against PyTorch autograd
on the same computation graph. Results match to floating-point precision.

---

### Mathematical proofs — `*.html`

Two rigorous derivations of the chain rule, written as interactive documents:

- **`chain_rule_demo.html`** — Multivariate chain rule: `L(f(x), g(x))`
  including the subtlety of the shifted evaluation point and the C¹ regularity requirement.

- **`chain_rule_single_function.html`** — Single-variable chain rule: `L(f(x))`
  derived from first principles using differentials.

---

## Key insight

The backpropagation algorithm is the multivariate chain rule, mechanized:

- Each node computes its **local gradient** (derivative of its output w.r.t. its inputs)
- The **upstream gradient** (`out.grad`) is multiplied in via the chain rule
- `+=` accumulates contributions from all paths through the graph
- The **topological sort** guarantees that upstream gradients are always
  fully computed before a node's `_backward` is called

```
self.grad += local_gradient * out.grad
```

This one line, applied in the right order, is all of backpropagation.

---

## Architecture

```
micrograd/
├── micro_grade.py              # Autograd engine (Value class)
├── MLP_object.py               # Neuron, Layer, MLP
├── train_demo.py               # Full training loop
├── pytorch_demo.py             # Gradient validation vs PyTorch
├── chain_rule_demo.html        # Multivariate chain rule proof
└── chain_rule_single_function.html  # Single-variable chain rule proof
```

---

## References

- Karpathy, A. — *The spelled-out intro to neural networks and backpropagation: building micrograd*
