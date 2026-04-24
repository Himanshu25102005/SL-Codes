# Generated from: 10_Ten_ANN.ipynb
# Converted at: 2026-04-24T03:49:19.833Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# ### ASSIGNMENT No: 10 
# 
# Title: Write Python program to implement CNN object detection. Discuss numerous performance evaluation metrics for evaluating the object detecting algorithms' performance. 
# 
# Problem Statement:  Implement any one of the following Expert System


import cv2 
import matplotlib.pyplot as plt 
from ultralytics import YOLO 
from collections import Counter 
# Load YOLO model 

model = YOLO("yolov8n.pt") 
print(" Model Loaded") 


# Image path (change accordingly) 
image_path = "./Image.jpg" 


# Read image 
img = cv2.imread(image_path) 

if img is None: 
    print(" Image NOT found. Fix path!") 
else: 
    print(" Image Loaded Successfully") 


# Run detection 
results = model(image_path) 
print(" Detection Done") 


# Extract detected objects 
detected_objects = [] 
for r in results: 
    for box in r.boxes: 
        cls = int(box.cls[0]) 
        label = model.names[cls] 
        detected_objects.append(label) 

print("Detected Objects:", detected_objects) 

# Convert BGR to RGB for display 
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

# Show image 
plt.imshow(img) 
plt.title("Original Image") 
plt.axis("off") 
plt.show()