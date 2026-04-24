# Generated from: 08_Eight_ANN.ipynb
# Converted at: 2026-04-24T03:48:06.563Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# ### ASSIGNMENT No: 08 
# 
# Title: Write a python program to illustrate ART neural network.
#  
# Problem Statement:  Implement  a python program to illustrate ART neural network.


import numpy as np

# 🔹 ART1 Class
class ART1:
    def __init__(self, vigilance=0.6):
        self.vigilance = vigilance
        self.clusters = []

    # 🔹 Similarity Function
    def similarity(self, x, y):
        return np.sum(np.minimum(x, y)) / np.sum(x)

    # 🔹 Training Function
    def fit(self, data):
        for i in range(len(data)):
            pattern = data[i]
            assigned = False
            
            # Compare with existing clusters
            for j in range(len(self.clusters)):
                sim = self.similarity(pattern, self.clusters[j])
                
                if sim >= self.vigilance:
                    # Update cluster (intersection)
                    self.clusters[j] = np.minimum(self.clusters[j], pattern)
                    assigned = True
                    break
            
            # If no match → create new cluster
            if not assigned:
                self.clusters.append(pattern.copy())

    # 🔹 Display Clusters
    def show_clusters(self):
        print("Clusters formed:\n")
        for i, c in enumerate(self.clusters):
            print(f"Cluster {i+1}: {c}")

# 🔹 Input Data
data = np.array([
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [0, 0, 1, 1],
    [0, 0, 1, 0]
])

# 🔹 Create Model
model = ART1(vigilance=0.6)

# 🔹 Train
model.fit(data)

# 🔹 Output
model.show_clusters()