# Generated from: 01_first_ann.ipynb
# Converted at: 2026-05-04T20:29:15.704Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# ### Assignment 1
# 
# Title: To study and understand the concept of Python program to plot a few activation functions that are being used in neural networks. 
# 
# Problem Statement : Write a Python program to plot a few activation functions that are being used in neural networks.


# !pip install numpy matplotlib

#(Lastest version upgrade)
# !pip install -U numpy matplotlib 

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)

def plot_activation(name, x, y):
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, color='blue', linewidth=2)
    plt.title(f"{name} Activation Function")
    plt.axhline(0, color='black', lw=1, alpha=0.5)
    plt.axvline(0, color='black', lw=1, alpha=0.5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def step(x):
    return np.where(x >= 0, 1, 0)

print("Plotting Step Function...")
plot_activation("Step", x, step(x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print("Plotting Sigmoid...")
plot_activation("Sigmoid", x, sigmoid(x))

def bipolar_sigmoid(x):
    return np.tanh(x)

print("Plotting Bipolar Sigmoid (Tanh)...")
plot_activation("Bipolar Sigmoid", x, bipolar_sigmoid(x))

def relu(x):
    return np.maximum(0, x)

print("Plotting ReLU...")
plot_activation("ReLU", x, relu(x))

def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)

print("Plotting Leaky ReLU...")
plot_activation("Leaky ReLU", x, leaky_relu(x))

def elu(x, alpha=1):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

print("Plotting ELU...")
plot_activation("ELU", x, elu(x))

def softmax(x):
    e_x = np.exp(x - np.max(x)) 
    return e_x / e_x.sum()

sample_inputs = np.array([2.0, 1.0, 0.1, -1.0, 3.5])
soft_outputs = softmax(sample_inputs)

plt.figure(figsize=(8, 4))
plt.bar(range(len(sample_inputs)), soft_outputs, color='purple', alpha=0.7)
plt.xticks(range(len(sample_inputs)), sample_inputs)
plt.title("Softmax Output (Probabilities for specific inputs)")
plt.xlabel("Input Value")
plt.ylabel("Probability")
plt.show()