# Generated from: 03_third_ann.ipynb
# Converted at: 2026-05-04T20:30:54.995Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# ### Problem 3
# 
# Title:  Implement digit recognition (even and odd numbers) using perceptron neural network 
# 
# Problem Statement: Write a Python Program using Perceptron Neural Network to recognize even and odd numbers. Given numbers are in ASCII from 0 to 9


# !pip install numpy scikit-learn matplotlib

# (Latest version upgrade)
# !pip install -U numpy scikit-learn matplotlib

# Import libraries
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Load dataset
digits = load_digits()
print(digits)

# Features and labels
X = digits.data
y = digits.target

X

y

# Convert to even (0) and odd (1)
y = np.where(y % 2 == 0, 0, 1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train perceptron model
model = Perceptron(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Show some predictions
for i in range(5):
    plt.imshow(X_test[i].reshape(8,8), cmap='gray')
    plt.title(f"Actual: {y_test[i]}  Predicted: {y_pred[i]}")
    plt.show()