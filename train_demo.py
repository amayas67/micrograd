from micro_grade import Value, draw_dot
from MLP_object import MLP
import random


# -------------------------------------------------------------------------
# Training demo: creating a loss function for a neural network
#
# A neural network does not learn directly from its predictions.
# We need a way to measure how far the predictions are from the desired
# targets. This measure is called the loss function.
# -------------------------------------------------------------------------


# Create a Multi-Layer Perceptron:
#
# Input layer: 3 features
# Hidden layer: 4 neurons
# Hidden layer: 4 neurons
# Output layer: 1 neuron
#
# At this point, the weights and biases are randomly initialized.
# The network has not learned anything yet.
n = MLP(3, [4, 4, 1])


# Training examples (inputs)
#
# Each element represents one sample.
#
# Example:
# [2.0, 3.0, -1.0]
#
# can be interpreted as three input features:
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
# Each input sample has a corresponding target value.
#
# Example:
#
# input:
# [2.0, 3.0, -1.0]
#
# should produce:
# 1.0
ys = [1.0, -1.0, -1.0, 1.0]


# Forward pass
#
# The network processes every input independently.
#
# Since each call to n(x) creates a computation graph,
# we can think of this as creating multiple graphs:
#
# Graph 1:
#
# x1 x2 x3
#  \ | /
#   MLP
#    |
#  prediction1
#
# Graph 2:
#
# x1 x2 x3
#  \ | /
#   MLP
#    |
#  prediction2

# And so on until we reach 4 graphs for our example
# Each graph shares the same MLP parameters (weights and biases).
ypred = [n(x) for x in xs]


# Compute the individual losses.
#
# For each prediction, we compare it with the desired target:
#
# loss_i = (prediction_i - target_i)^2
#
# Example:
#
# prediction = 0.8
# target = 1.0
#
# loss = (0.8 - 1.0)^2
# loss = 0.04
#
# A smaller value means that the prediction is closer to the target.
loss = [(prediction - target)**2 for prediction, target in zip(ypred, ys)]


# At this point we have several independent losses:
#
# loss = [
#     loss1,
#     loss2,
#     loss3,
#     loss4
# ]
#
# However, backpropagation needs a single scalar objective.
#
# Therefore, we combine all individual losses into one global loss:
#
# Loss = loss1 + loss2 + loss3 + loss4
#
# This creates one final computation graph connecting all examples.
loss = sum(loss)


print(loss)
