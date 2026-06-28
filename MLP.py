
class Neuron:
    """
    Fully connected neuron.

    Computes:
        tanh(w·x + b)
    """

    def __init__(self, n):
        # One learnable weight per input feature.
        self.W = [Value(random.uniform(-1, 1)) for _ in range(n)]

        # Learnable bias.
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, X):
        # Weighted sum of the inputs:
        # w1*x1 + w2*x2 + ... + wn*xn + b
        act = sum((wi * xi for wi, xi in zip(self.W, X)), self.b)

        # Non-linear activation.
        return act.tanh()


class Layer:
    """
    Fully connected layer.

    A layer is simply a collection of independent neurons.
    Every neuron receives the same input vector and produces
    its own activation.
    """

    def __init__(self, nin, nout):
        # Create 'nout' neurons, each expecting 'nin' inputs.
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [neuron(x) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out

class MLP:
    """
    Multi-Layer Perceptron (MLP).

    An MLP is a sequence of fully connected layers.
    The output of one layer becomes the input of the next.
    """

    def __init__(self, nin, nouts):
        # Example:
        # nin = 3
        # nouts = [4, 4, 1]
        #
        # sz = [3, 4, 4, 1]
        #
        # Layers created:
        # Layer(3,4) -> Layer(4,4) -> Layer(4,1)
        sz = [nin] + nouts

        self.layers = [
            Layer(sz[i], sz[i + 1])
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        # Forward propagation.
        # Each layer receives the output of the previous one.
        for layer in self.layers:
            x = layer(x)

        return x





# -------------------------------------------------------------------------
# Example: forward pass through a small MLP
# -------------------------------------------------------------------------

# Three input features
x = [Value(2.0), Value(3.0), Value(-1.0)]

# MLP architecture:
# 3 inputs -> 4 neurons -> 4 neurons -> 1 output
mlp = MLP(3, [4, 4, 1])

# Forward propagation
out = mlp(x)

print(out)

# Compute gradients for the whole network
out.cal_backward()

# Visualize the complete computation graph
dot = draw_dot(out)
dot.render("mlp_computation_graph", view=False)
print("Graph saved to mlp_computation_graph.svg")
