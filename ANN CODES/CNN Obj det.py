# Generated from: CNN Obj det.ipynb
# Converted at: 2026-05-05T02:08:18.295Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# ### ASSIGNMENT No: 10 
# 
# Title: Write Python program to implement CNN object detection. Discuss numerous performance evaluation metrics for evaluating the object detecting algorithms' performance. 
# 
# Problem Statement:  Implement any one of the following Expert System


# !pip install ultralytics opencv-python matplotlib numpy

# (Latest version upgrade)
# !pip install -U ultralytics opencv-python matplotlib numpy

#Optional (only if error comes)
## !pip install torch torchvision

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt") 
print(" Model Loaded") 

# Image path (change accordingly or get an image and rename it Image.jpg and keep it in the same directory as this notebook) 
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

annotated_img = results[0].plot()

plt.imshow(annotated_img)
plt.title("Detected Objects (YOLO)")
plt.axis("off")
plt.show()