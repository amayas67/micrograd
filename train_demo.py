from micro_grade import Value, draw_dot
from MLP_object import MLP
import random


def recalculate(model, xs, ys):
    """
    Perform a new forward pass using the current model parameters.

    After gradient descent updates the weights, the previous computation
    graph becomes outdated. Therefore, we rebuild the graph by computing
    new predictions and a new global loss.

    Returns:
        ypred : list of network predictions
        loss  : scalar loss for the current model parameters
    """

    # Forward pass
    ypred = [model(x) for x in xs]

    # Global loss
    loss = sum((pred - target) ** 2 for pred, target in zip(ypred, ys))

    return ypred, loss


# -------------------------------------------------------------------------
# Training demo: creating a loss function for a neural network
#
# A neural network does not learn directly from its predictions.
# We first need a way to measure how good or bad its predictions are.
# This measure is called the loss function.
# -------------------------------------------------------------------------


# Create a Multi-Layer Perceptron:
#
# Input layer : 3 features
# Hidden layer: 4 neurons
# Hidden layer: 4 neurons
# Output layer: 1 neuron
#
# At this point every weight and bias is initialized randomly.
# The network has not learned anything yet.
n = MLP(3, [4, 4, 1])


# Training examples
#
# Each element is one independent training sample.
#
# Example:
#
# [2.0, 3.0, -1.0]
#
# can be interpreted as:
#
# x1 = 2.0
# x2 = 3.0
# x3 = -1.0
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]


# Desired outputs (targets)
#
# Every training example has one expected prediction.
ys = [1.0, -1.0, -1.0, 1.0]


# -------------------------------------------------------------------------
# Forward pass
#
# Each call to n(x) builds a new computation graph.
#
# We can visualize the process as:
#
# Graph 1:
#
# x1 x2 x3
#  \ | /
#   MLP
#    |
# prediction1
#
# Graph 2:
#
# x1 x2 x3
#  \ | /
#   MLP
#    |
# prediction2
#
# ...
#
# Although four different graphs are created, they all share the exact
# same network parameters (weights and biases).
# -------------------------------------------------------------------------
ypred = [n(x) for x in xs]


# -------------------------------------------------------------------------
# Compute the individual losses.
#
# For each prediction:
#
#     loss_i = (prediction_i - target_i)^2
#
# Example:
#
# prediction = 0.8
# target     = 1.0
#
# loss = (0.8 - 1.0)^2 = 0.04
#
# Smaller losses correspond to better predictions.
# -------------------------------------------------------------------------
loss = [(prediction - target) ** 2 for prediction, target in zip(ypred, ys)]


# -------------------------------------------------------------------------
# Backpropagation requires a single scalar objective.
#
# Therefore we combine all individual losses into one global loss:
#
# Loss = loss1 + loss2 + loss3 + loss4
#
# This creates one final computation graph connecting every training
# example to the shared network parameters.
# -------------------------------------------------------------------------
loss = sum(loss)

print("Loss before training:", loss)


# -------------------------------------------------------------------------
# Backpropagation
#
# Compute the gradient of the global loss with respect to every parameter
# of the network.
# -------------------------------------------------------------------------
loss.cal_backward()

# -------------------------------------------------------------------------
# Gradient descent step
#
# Every parameter already stores its gradient.
#
# We update each parameter by moving it in the opposite direction of its
# gradient:
#
#     parameter = parameter - learning_rate * gradient
#
# The learning rate controls the size of each update:
#
# - A large learning rate speeds up learning but may overshoot the minimum,
#   making training unstable or even causing divergence.
#   for example
# A parameter may receive a large positive update because its gradient is
# initially negative. After the update, its gradient can become positive,
# meaning the next step will immediately start decreasing that parameter.
# A smaller learning rate would have produced a more gradual adjustment.
# - A small learning rate is usually more stable but requires many more
#   iterations to converge.
#
# Choosing an appropriate learning rate is therefore one of the most
# important aspects of training a neural network.
# -------------------------------------------------------------------------
learning_rate = 0.01

for epoch in range(1000):
    ypred, loss = recalculate(n, xs, ys)
    loss.cal_backward()
    # No need to manually reset the gradients.
# cal_backward() automatically clears every gradient before computing
# the new backward pass.
    for p in n.parameters():
        p.data +=  -learning_rate * p.grad
    if epoch % 10 == 0:
        print(f"epoch {epoch}, loss: {loss.data:.4f}")

# The loss should now be smaller than before, showing that one gradient
# descent step improved the model.
print("Loss after gradient descent:", loss)
print("predictions after  gradient descent:", ypred)



# Uncomment to visualize the final computation graph.
#
# dot = draw_dot(loss)
# dot.render("mlp_computation_graph_2", view=False)
# print("Graph saved to mlp_computation_graph_2.svg")
