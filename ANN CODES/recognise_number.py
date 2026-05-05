# Generated from: 05_five_ann.ipynb
# Converted at: 2026-05-04T20:32:41.767Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# ### Assignment 5
# 
# Title: To illustrate Artificial Neural Network training process in Python
# 
# Problem Statement: Implement Artificial Neural Network training process in Python by using Forward Propagation, Back Propagation


# !pip install numpy

# (Latest version upgrade)
# !pip install -U numpy

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])

np.random.seed(42)

input_neurons = 2
hidden_neurons = 3
output_neurons = 1

W1 = np.random.uniform(size=(input_neurons, hidden_neurons))
b1 = np.random.uniform(size=(1, hidden_neurons))

W2 = np.random.uniform(size=(hidden_neurons, output_neurons))
b2 = np.random.uniform(size=(1, output_neurons))

learning_rate = 0.1
epochs = 10000

for epoch in range(epochs):
    
    # ---- Forward Propagation ---
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)
    
    final_input = np.dot(hidden_output, W2) + b2
    predicted_output = sigmoid(final_input)
    
    # ---- Error Calculation ---
    error = y - predicted_output
    
    # ---- Backpropagation ---
    d_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)
    
    # ---- Update Weights ---
    W2 += hidden_output.T.dot(d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    
    W1 += X.T.dot(d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

print("\nFinal Predicted Output:")
predicted_output

print("\nRounded Predictions (0 or 1):")
np.round(predicted_output).astype(int)

print("\nExpected Output:")
y