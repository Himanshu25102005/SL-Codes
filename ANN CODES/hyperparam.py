# Generated from: hyperparam.ipynb
# Converted at: 2026-05-05T02:08:14.704Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# ### ASSIGNMENT No: 11
# 
# Title: For an image classification challenge, create and train a ConvNet in Python using TensorFlow. Also, try to improve the performance of the model by applying various hyper parameter tuning to reduce the overfitting or under fitting problem that might occur. Maintain graphs of comparisons.
# 
# Problem Statement:  Implement an image classification challenge, create and train a ConvNet in Python using TensorFlow 


# !pip install numpy matplotlib tensorflow opencv-python

# (Latest version upgrade)
# !pip install -U numpy matplotlib tensorflow opencv-python

#If tensorflow installation takes time
# !pip install tensorflow-cpu

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
# from tensorflow.keras.datasets import fashion_mnist (If you are using fashion Mnist uncomment this and delete cifar10)


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data() (If you are using fashion Mnist uncomment this and delete cifar10)

# Normalize data (0–255 → 0–1)
x_train = x_train / 255.0
x_test = x_test / 255.0

print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)

# Use this only when fashion mnist
# x_train = x_train.reshape(-1, 28, 28, 1)
# x_test = x_test.reshape(-1, 28, 28, 1)

def create_model(filters=32, dropout_rate=0.0):
    model = models.Sequential([
        layers.Conv2D(filters, (3,3), activation='relu', input_shape=(32,32,3)),
        # layers.Conv2D(filters, (3,3), activation='relu', input_shape=(28,28,1)), (for fashion MNIST) (Delete above layer if using fashion MNIST)
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(filters*2, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        
        layers.Dropout(dropout_rate),  # helps reduce overfitting
        
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

model1 = create_model(filters=32, dropout_rate=0.0)

history1 = model1.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

model2 = create_model(filters=32, dropout_rate=0.5)

history2 = model2.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

print("Without Dropout Accuracy:", history1.history['val_accuracy'][-1])
print("With Dropout Accuracy:", history2.history['val_accuracy'][-1])

plt.figure(figsize=(10,4))

# Accuracy comparison
plt.subplot(1,2,1)
plt.plot(history1.history['val_accuracy'], label='Without Dropout')
plt.plot(history2.history['val_accuracy'], label='With Dropout')
plt.title("Validation Accuracy")
plt.legend()

# Loss comparison
plt.subplot(1,2,2)
plt.plot(history1.history['val_loss'], label='Without Dropout')
plt.plot(history2.history['val_loss'], label='With Dropout')
plt.title("Validation Loss")
plt.legend()

plt.show()

loss, acc = model2.evaluate(x_test, y_test)
print("Test Accuracy:", acc)

import cv2
import numpy as np

# Image path (keep image.jpg in same folder)
img = cv2.imread("image.jpg")

# -------------------------------
# 🔹 For CIFAR-10 (RGB images)
# -------------------------------
img = cv2.resize(img, (32, 32))  # CIFAR uses 32x32 size
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR → RGB
img = img / 255.0  # Normalize

# -------------------------------
# 🔹 For Fashion-MNIST (Grayscale)
# -------------------------------
# Uncomment below and comment CIFAR part if using Fashion-MNIST
# img = cv2.resize(img, (28, 28))           # Fashion MNIST uses 28x28
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
# img = img / 255.0
# img = img.reshape(28, 28, 1)              # Add channel dimension

# Add batch dimension (same for both)
img = np.expand_dims(img, axis=0)

# Predict
prediction = model2.predict(img)

# -------------------------------
# 🔹 Class Names
# -------------------------------
# CIFAR-10 classes (currently active)
class_names = ['Airplane','Automobile','Bird','Cat','Deer',
               'Dog','Frog','Horse','Ship','Truck']

# For Fashion-MNIST (uncomment if using it)
# class_names = ['T-shirt','Trouser','Pullover','Dress','Coat',
#                'Sandal','Shirt','Sneaker','Bag','Ankle boot']

# Get predicted class
pred_class = class_names[np.argmax(prediction)]

print("Predicted Class:", pred_class)

plt.imshow(img[0])
plt.title(f"Prediction: {pred_class}")
plt.axis('off')
plt.show()