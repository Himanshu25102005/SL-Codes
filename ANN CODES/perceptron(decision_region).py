# Generated from: 04_four_ann.ipynb
# Converted at: 2026-05-04T20:31:38.025Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# ### Assignment 4
# 
# Title: To study and understand the concept to recognize the numbers 0, 1, 2, …9. A 5 * 3 matrix forms. 
# 
# Problem Statement: Write a python program to recognize the numbers 0, 1, 2, ..9. A 5 * 3 matrix forms the numbers. For any valid point it is taken as 1 and invalid 
# point it is taken as 0. The net has to be trained to recognise all the numbers and when the test data is given, the network has to recognise the particular numbers 


# !pip install numpy matplotlib

# (Latest version upgrade)
# !pip install -U numpy matplotlib

import numpy as np
import matplotlib.pyplot as plt

# 1. Data Setup (5x3 digit patterns)
X = np.array([
    [1,1,1, 1,0,1, 1,0,1, 1,0,1, 1,1,1],  # 0
    [0,1,0, 1,1,0, 0,1,0, 0,1,0, 1,1,1],  # 1
    [1,1,1, 0,0,1, 1,1,1, 1,0,0, 1,1,1],  # 2
    [1,1,1, 0,0,1, 1,1,1, 0,0,1, 1,1,1],  # 3
    [1,0,1, 1,0,1, 1,1,1, 0,0,1, 0,0,1],  # 4
    [1,1,1, 1,0,0, 1,1,1, 0,0,1, 1,1,1],  # 5
    [1,1,1, 1,0,0, 1,1,1, 1,0,1, 1,1,1],  # 6
    [1,1,1, 0,0,1, 0,0,1, 0,0,1, 0,0,1],  # 7
    [1,1,1, 1,0,1, 1,1,1, 1,0,1, 1,1,1],  # 8
    [1,1,1, 1,0,1, 1,1,1, 0,0,1, 1,1,1]   # 9
])

# Function to visualize patterns
def display_patterns(data, titles, rows=2, cols=5):
    plt.figure(figsize=(12, 6))
    for i in range(len(data)):
        plt.subplot(rows, cols, i + 1)
        # Reshape the 15 features into a 5x3 grid
        grid = data[i].reshape(5, 3)
        plt.imshow(grid, cmap='binary')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Visualize training data
print("Visualizing Training Set:")
display_patterns(X, [f"Digit {i}" for i in range(10)])

# 2. Perceptron training function (One-vs-All)
def train_perceptron(X, y_binary, lr=0.1, epochs=100):
    w = np.zeros(X.shape[1])
    b = 0
    for _ in range(epochs):
        for i in range(len(X)):
            z = np.dot(X[i], w) + b
            y_pred = 1 if z >= 0 else -1
            if y_pred != y_binary[i]:
                w += lr * y_binary[i] * X[i]
                b += lr * y_binary[i]
    return w, b

# Train models
models = []
for digit in range(10):
    y_binary = np.where(np.arange(10) == digit, 1, -1)
    w, b = train_perceptron(X, y_binary)
    models.append((w, b))

# 3. Prediction function
def predict(x, models):
    scores = [np.dot(x, w) + b for w, b in models]
    return np.argmax(scores)

# Test on training patterns
print("\nTesting all digit patterns:")
for i in range(10):
    pred = predict(X[i], models)
    status = "✓" if pred == i else "✗"
    print(f"Input: {i} | Predicted: {pred} | {status}")

# 4. Noisy Input Test & Visualization
noisy_0 = np.array([1,1,1, 1,0,1, 1,0,0, 1,0,1, 1,1,1])  
print(f"\nNoisy input predicted as: {predict(noisy_0, models)}")

plt.figure(figsize=(3, 3))
plt.imshow(noisy_0.reshape(5, 3), cmap='Reds') # Using Red to highlight noise
plt.title("Noisy '0' Input")
plt.axis('off')
plt.show()