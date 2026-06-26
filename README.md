micrograd

    From scratch autograd engine — understand backpropagation, don't just use it.

A minimal, educational implementation of automatic differentiation (reverse-mode autograd) in pure Python, inspired by Andrej Karpathy's micrograd.
No PyTorch. No TensorFlow. Just Python, math, and the chain rule.
Why this repo?
Most people learn deep learning by calling loss.backward() and hoping for the best.
This repo is the antidote: every gradient is computed by hand, every topological sort is built from scratch, and every operation traces its own local derivative.
If you want to stop being a user of black boxes and start understanding why neural networks learn, start here.
What you get
Core engine (micro_grade.py)
Table
Feature	Description
Value class	Scalar tensor with autograd support
+, *, -, /, **	Basic arithmetic with full gradient tracking
tanh(), exp()	Activation functions with local derivatives
topological_order_reversed()	Post-order DFS topological sort for correct backprop
cal_backward()	Reset-safe backward pass (output → inputs)
draw_dot()	Graphviz visualization of the computation graph
Mathematical proofs (HTML)
Table
File	Content
chain_rule_demo.html	Multivariate chain rule — the subtlety of shifted evaluation points and C1  continuity
demo_single_fonction_chain.html	Single-variable chain rule — why arepsilon = f'(x) \cdot dx  is valid
These are not copy-pasted from a textbook. They are step-by-step derivations written to justify every line of code in micro_grade.py.
Validation test
The repo includes a numerical validation that compares two implementations of anh(x) :

    Direct: i.tanh()
    Via identity: e2x+1e2x−1​ 

Both paths produce identical gradients (within floating-point precision, ∼10−15 ), proving the engine is consistent.
PyTorch parity check (pytorch_demo.py)
A side-by-side comparison with PyTorch on the same computation graph. Our Value gradients match PyTorch's Tensor.grad exactly.
Quick start
Python

from micro_grade import Value

# Build a computation graph
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')

d = a * b        # -6
e = d + c        # 4
f = e.tanh()     # tanh(4)

# Backpropagate
f.cal_backward()

# Inspect gradients
print(f"a.grad = {a.grad}")   # ∂f/∂a
print(f"b.grad = {b.grad}")   # ∂f/∂b
print(f"c.grad = {c.grad}")   # ∂f/∂c

Visualize the graph:
Python

from micro_grade import draw_dot
dot = draw_dot(f)
dot.render("graph", view=True)

The chain rule, in code
Every _backward() closure is a direct translation of the chain rule.
For multiplication out = self * other:
Python

def _backward():
    self.grad  += other.data * out.grad   # dL/dself = dL/dout * dout/dself
    other.grad += self.data  * out.grad   # dL/dother = dL/dout * dout/dother

The += (not =) is the multivariate chain rule in action: gradients accumulate when a node is used in multiple paths.
Project structure
plain

micrograd/
├── micro_grade.py              # Core autograd engine
├── pytorch_demo.py             # Parity check with PyTorch
├── chain_rule_demo.html        # Multivariate chain rule proof
├── demo_single_fonction_chain.html  # Single-variable chain rule proof
└── computation_graph_complex.svg    # Example visualization

Requirements

    Python 3.8+
    graphviz (system package + Python package)
    numpy, matplotlib (for visualization)
    torch (only for pytorch_demo.py)

bash

pip install graphviz numpy matplotlib torch

License
MIT — do whatever you want, but understand it first.

    "I have no special talent. I am only passionately curious." — Albert Einstein
