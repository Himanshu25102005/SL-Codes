# Generated from: Iris_Flower.ipynb
# Converted at: 2026-04-23T03:19:32.666Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# Title: Data Visualization III 


# 
# PROBLEM STATEMENT: 
#  
# Download the Iris flower dataset or any other dataset into a DataFrame.(e.g., 
# https://archive.ics.uci.edu/ml/datasets/Iris). Scan the dataset and give the inference as: 
# 1. List down the features and their types (e.g., numeric, nominal) available in the dataset. 
#  
# 2. Create a histogram for each feature in the dataset to illustrate the feature distributions. 
#  
# 3. Create a boxplot for each feature in the dataset. 
#  
# 4. Compare distributions and identify outliers. 


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')

iris.head()

print("Features and their data types:\n")
print(iris.dtypes)

iris.hist(figsize=(10,8))
plt.suptitle("Histograms of Iris Features")
plt.show()

numeric_cols = iris.select_dtypes(include=['float64', 'int64']).columns

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(y=iris[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

for col in numeric_cols:
    Q1 = iris[col].quantile(0.25)
    Q3 = iris[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outliers = iris[(iris[col] < lower) | (iris[col] > upper)]
    
    print(f"{col} - Number of outliers:", len(outliers))

# Summary statistics for comparison
print("Statistical Summary:\n")
print(iris.describe())

# Compare spread using standard deviation
print("\nStandard Deviation (Spread of Features):")
print(iris.std(numeric_only=True))

# Compare median values
print("\nMedian Values:")
print(iris.median(numeric_only=True))

# Observation: Petal features have lower spread and better separation compared to sepal features.