# Generated from: 09_Nine_ANN.ipynb
# Converted at: 2026-04-24T03:48:38.546Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# ### ASSIGNMENT No: 09 
# 
# Title: Write a python program to design a Hopfield Network which stores 4 vectors  
# 
# Problem Statement:  Design a Hopfield Network which stores 4 vectors  


import numpy as np

patterns = np.array([
    [1, -1, 1, -1],
    [-1, 1, -1, 1],
    [1, 1, -1, -1],
    [-1, -1, 1, 1]
])

class HopfieldNetwork:
    
    def __init__(self):
        self.W = None   # weight matrix

    # 🔹 Train using Hebbian Learning
    def train(self, patterns):
        n = patterns.shape[1]
        self.W = np.zeros((n, n))
        
        for p in patterns:
            p = p.reshape(n, 1)
            self.W += p @ p.T   # outer product
        
        # remove self connections
        np.fill_diagonal(self.W, 0)

    # 🔹 Recall function
    def recall(self, x, steps=5):
        x = x.copy()
        
        for _ in range(steps):
            x = np.sign(self.W @ x)
        
        return x

model = HopfieldNetwork()
model.train(patterns)

print("Weight Matrix:\n", model.W)

# slightly corrupted pattern
test = np.array([1, -1, 1, -1])

print("Input:", test)
print("\n")

output = model.recall(test)

print("Recovered Output:", output)