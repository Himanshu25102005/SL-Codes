# Generated from: backprop XOR.ipynb
# Converted at: 2026-05-05T02:08:31.663Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# ### Assignment 7
# 
# Title: Write a python program to show Back Propagation Network for XOR function with Binary Input and Output 
# 
# Problem Statement: Implement to show Back Propagation Network for XOR function with Binary Input and Output


# !pip install numpy matplotlib

# (Latest version upgrade)
# !pip install -U numpy matplotlib

import numpy as np
import matplotlib.pyplot as plt

# XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0])

class SimpleNN:
    def __init__(self, lr=0.5, epochs=10000):
        self.lr = lr
        self.epochs = epochs
        
        np.random.seed(1)
        
        # Weights (input → hidden)
        self.w1 = np.random.rand()
        self.w2 = np.random.rand()
        self.w3 = np.random.rand()
        self.w4 = np.random.rand()
        
        # Weights (hidden → output)
        self.w5 = np.random.rand()
        self.w6 = np.random.rand()
        
        # Bias
        self.b1 = np.random.rand()
        self.b2 = np.random.rand()
        
        self.losses = []

    # Activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # Derivative of sigmoid
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # Forward pass
    def forward(self, x1, x2):
        # Hidden layer
        h1_input = x1 * self.w1 + x2 * self.w2 + self.b1
        h1 = self.sigmoid(h1_input)

        h2_input = x1 * self.w3 + x2 * self.w4 + self.b1
        h2 = self.sigmoid(h2_input)

        # Output layer
        output_input = h1 * self.w5 + h2 * self.w6 + self.b2
        output = self.sigmoid(output_input)

        return h1, h2, output

    # Training function
    def fit(self, X, y):
        for epoch in range(self.epochs):
            total_loss = 0
            
            for i in range(len(X)):
                x1 = X[i][0]
                x2 = X[i][1]
                target = y[i]
                
                # ---- Forward ----
                h1, h2, output = self.forward(x1, x2)
                
                # ---- Error ----
                error = target - output
                total_loss = total_loss + (error ** 2)
                
                # ---- Backprop ----
                d_output = error * self.sigmoid_derivative(output)
                
                d_h1 = d_output * self.w5 * self.sigmoid_derivative(h1)
                d_h2 = d_output * self.w6 * self.sigmoid_derivative(h2)
                
                # ---- Update output layer ----
                self.w5 = self.w5 + self.lr * d_output * h1
                self.w6 = self.w6 + self.lr * d_output * h2
                self.b2 = self.b2 + self.lr * d_output
                
                # ---- Update hidden layer ----
                self.w1 = self.w1 + self.lr * d_h1 * x1
                self.w2 = self.w2 + self.lr * d_h1 * x2
                
                self.w3 = self.w3 + self.lr * d_h2 * x1
                self.w4 = self.w4 + self.lr * d_h2 * x2
                
                # Bias for hidden layer
                self.b1 = self.b1 + self.lr * (d_h1 + d_h2)
            
            self.losses.append(total_loss)

    def predict(self, X):
        results = []
        
        for i in range(len(X)):
            x1 = X[i][0]
            x2 = X[i][1]
            
            _, _, output = self.forward(x1, x2)
            
            # Convert to 0 or 1
            if output >= 0.5:
                results.append(1)
            else:
                results.append(0)
                
        return results

    def plot_loss(self):
        plt.plot(self.losses)
        plt.title("Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

model = SimpleNN(lr=0.5, epochs=10000)
model.fit(X, y)

model.plot_loss()

print("Final Predictions:\n")

preds = model.predict(X)

for i in range(len(X)):
    print(X[i], "->", preds[i])