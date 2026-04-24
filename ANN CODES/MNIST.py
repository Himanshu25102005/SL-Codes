# Generated from: 13_Thirteen.ipynb
# Converted at: 2026-04-24T03:51:16.053Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# ### ASSIGNMENT No: 13 
# 
# Title: MNIST Handwritten Character Detection using PyTorch, Keras and Tensorflow 
# 
# Problem Statement:  Implement MNIST Handwritten Character Detection using PyTorch, Keras and Tensorflow 


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for CNN
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

# Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Test
loss, acc = model.evaluate(x_test, y_test)
print("Accuracy:", acc)

import random

predictions = model.predict(x_test)

plt.figure(figsize=(6,6))

for i in range(9):
    idx = random.randint(0, len(x_test)-1)
    
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[idx].reshape(28,28), cmap='gray')
    
    pred = np.argmax(predictions[idx])
    actual = y_test[idx]
    
    color = "green" if pred == actual else "red"
    plt.title(f"P:{pred} A:{actual}", color=color)
    
    plt.axis('off')

plt.suptitle("TensorFlow Predictions (Random Images)")
plt.show()

# ## Keras Only Implementation


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.datasets import mnist

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0



# Reshape
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)



# Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])



model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



model.fit(x_train, y_train, epochs=5)

import random

random.seed(42)  # different selection

predictions = model.predict(x_test)

plt.figure(figsize=(6,6))

for i in range(9):
    idx = random.randint(0, len(x_test)-1)
    
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[idx].reshape(28,28), cmap='gray')
    
    pred = np.argmax(predictions[idx])
    actual = y_test[idx]
    
    color = "green" if pred == actual else "red"
    plt.title(f"P:{pred} A:{actual}", color=color)
    
    plt.axis('off')

plt.suptitle("Keras Predictions (Random Images)")
plt.show()

# ## PyTorch Implementation


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Load dataset
transform = transforms.ToTensor()

# 🔹 Input Data
data = np.array([
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [0, 0, 1, 1],
    [0, 0, 1, 0]
])

train_data = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_data = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(5408, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(-1, 5408)
        x = self.fc(x)
        return x


model = CNN()

# Loss + Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training (1 epoch only for simplicity)
for epoch in range(1):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print("Training done")

# Get shuffled batch
test_loader_iter = iter(test_loader)
images, labels = next(test_loader_iter)

outputs = model(images)
_, preds = torch.max(outputs, 1)

plt.figure(figsize=(6,6))

for i in range(9):
    idx = random.randint(0, len(images)-1)
    
    plt.subplot(3,3,i+1)
    plt.imshow(images[idx].squeeze(), cmap='gray')
    
    pred = preds[idx].item()
    actual = labels[idx].item()
    
    color = "green" if pred == actual else "red"
    plt.title(f"P:{pred} A:{actual}", color=color)
    
    plt.axis('off')

plt.suptitle("PyTorch Predictions (Random Images)")
plt.show()