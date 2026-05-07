# Import libraries

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# -------------------------------

# 1. Load Dataset

# -------------------------------

digits = load_digits()

# Features (images flattened into vectors)

X = digits.data

# Labels (0–9 digits)

y = digits.target

# -------------------------------

# 2. Convert to Even (0) / Odd (1)

# -------------------------------

y = np.where(y % 2 == 0, 0, 1)

# -------------------------------

# 3. Split Data

# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

# -------------------------------

# 4. Train Perceptron Model

# -------------------------------

model = Perceptron(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------------

# 5. Predictions

# -------------------------------

y_pred = model.predict(X_test)

# -------------------------------

# 6. Evaluation

# -------------------------------

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------------

# 7. Show Sample Predictions

# -------------------------------

for i in range(5):
    plt.imshow(X_test[i].reshape(8, 8), cmap='gray')
    plt.title(f"Actual: {y_test[i]}  Predicted: {y_pred[i]}")
    plt.axis('off')
    plt.show()

# -------------------------------

# 8. (Optional) View Learned Weights

# -------------------------------

print("Weights shape:", model.coef_.shape)
print("Bias:", model.intercept_)
