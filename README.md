# micrograd-like Autograd Engine

This project is a **from-scratch implementation of a simple automatic differentiation engine**, inspired by Andrej Karpathy’s micrograd.

It demonstrates how modern deep learning frameworks like PyTorch implement **backpropagation through a computation graph**.

---

## 🎯 Objective

The goal of this project is to deeply understand:

- How computation graphs are built dynamically
- How the chain rule is applied during backpropagation
- How gradients flow through operations
- How frameworks like PyTorch implement autograd internally

This is a **pedagogical implementation**, not an optimized deep learning library.

---

## ⚙️ Features

- Scalar-valued automatic differentiation (`Value` class)
- Support for basic operations:
  - addition `+`
  - multiplication `*`
  - power `**`
  - division `/`
  - negation `-`
  - exponential `exp`
  - hyperbolic tangent `tanh`
- Reverse-mode autodiff (backpropagation)
- Topological sorting of computation graph
- Manual gradient computation via `_backward()` functions
- Graph visualization using Graphviz

---

## 🧠 Core Idea

Each operation creates a node in a computation graph.

Example:
a ----*
       \
        (*) ---- +
b ----*        |
               d


```python
a = Value(2.0)
b = Value(-3.0)
c = a * b
d = c + 1
d.cal_backward()

