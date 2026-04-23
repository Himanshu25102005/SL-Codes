# Generated from: 06_Exp6.ipynb
# Converted at: 2026-04-23T03:09:34.888Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# Title: Data Analytics III


# PROBLEM STATEMENT: 
#  
# 1. Implement Simple Naïve Bayes classification algorithm using Python/R on iris.csv dataset. 
#  
# 2. Compute Confusion matrix to find TP, FP, TN, FN, Accuracy, Error rate, Precision, Recall on the given 
# dataset.


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n", cm)

for i in range(len(class_names)):
    TP = cm[i][i]
    FP = sum(cm[:, i]) - TP
    FN = sum(cm[i, :]) - TP
    TN = np.sum(cm) - (TP + FP + FN)
    
    print(f"\nClass: {class_names[i]}")
    print("TP:", TP)
    print("FP:", FP)
    print("TN:", TN)
    print("FN:", FN)

accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy

print("\nAccuracy:", accuracy)
print("Error Rate:", error_rate)

precision_list = []
recall_list = []

for i in range(len(class_names)):
    TP = cm[i][i]
    FP = sum(cm[:, i]) - TP
    FN = sum(cm[i, :]) - TP
    
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    
    precision_list.append(precision)
    recall_list.append(recall)

precision_macro = np.mean(precision_list)
recall_macro = np.mean(recall_list)

print("Precision (macro):", precision_macro)
print("Recall (macro):", recall_macro)