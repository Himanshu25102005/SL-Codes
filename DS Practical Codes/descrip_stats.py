# Generated from: 03_Exp3.ipynb
# Converted at: 2026-05-06T07:17:00.554Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# Title: Data Wrangling II


# 
# Descriptive Statistics - Measures of Central Tendency and variability perform the following operations on any 
# open source dataset (e.g., data.csv) 
# 1.Provide summary statistics (mean, median, minimum, maximum, standard deviation) for a dataset (age, 
# income etc.) with numeric variables grouped by one of the qualitative (categorical) variables. For example, 
# if your categorical variable is age groups and quantitative variable is income, then provide summary 
# statistics of income grouped by the age groups. Create a list that contains a numeric value for each response 
# to the categorical variable. 
# 2. Write a Python program to display some basic statistical details like percentile, mean, standard deviation 
# etc. of the species of ‘Iris-setosa’, ‘Iris-versicolor’ and ‘Iris-versicolor’ of iris.csv dataset. 


import pandas as pd
import numpy as np

url_credit = 'https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Credit.csv'
df_credit = pd.read_csv(url_credit, index_col=0)
df_credit

# Create categorical 'Age_Group' variable by binning the numerical 'Age'
bins = [20, 30, 40, 50, 60, 70, 80, 90, 100]
labels = ['20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
df_credit['Age_Group'] = pd.cut(df_credit['Age'], bins=bins, labels=labels, right=False)

# Calculate summary statistics of numeric 'Income' grouped by the categorical 'Age_Group'
summary_stats = df_credit.groupby('Age_Group', observed=False)['Income'].agg(['mean', 'median', 'min', 'max', 'std'])
display(summary_stats)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df_credit['Age_Group_Numeric'] = le.fit_transform(df_credit['Age_Group'].astype(str))
numeric_responses = df_credit['Age_Group_Numeric'].tolist()

print("First 30 numeric values mapped to categorical 'Age_Group' response:\n", numeric_responses[:30])

# ### Task 2: Basic Statistical Details of Iris Species


url_iris = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'
df_iris = pd.read_csv(url_iris)

setosa_stats = df_iris[df_iris['species'] == 'setosa'].describe()
versicolor_stats = df_iris[df_iris['species'] == 'versicolor'].describe()
virginica_stats = df_iris[df_iris['species'] == 'virginica'].describe()

print("--- Iris-setosa Statistics ---")
display(setosa_stats)

print("\n--- Iris-versicolor Statistics ---")
display(versicolor_stats)

print("\n--- Iris-virginica Statistics ---")
display(virginica_stats)