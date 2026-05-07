# Generated from: 01_Exp1.ipynb
# Converted at: 2026-05-06T07:17:04.796Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# Title: Data Wrangling I 


#  
# PROBLEM STATEMENT: 
#  
# Perform the following operations using Python on any open source dataset (e.g., data.csv) 
# 1. Import all the required Python Libraries. 
# 2. Locate an open source data from the web (e.g., https://www.kaggle.com). Provide a clear description of the 
# data and its source (i.e., URL of the web site). 
# 3. Load the Dataset into pandas dataframe. 
# 4. Data Preprocessing: check for missing values in the data using pandas isnull(), describe() function to get 
# some initial statistics. Provide variable descriptions. Types of variables etc. Check the dimensions of the 
# data frame. 
# 5. Data Formatting and Data Normalization: Summarize the types of variables by checking the data types (i.e., 
# character, numeric, integer, factor, and logical) of the variables in the data set. If variables are not in the 
# correct data type, apply proper type conversions. 
# 6. Turn categorical variables into quantitative variables in Python. 


### 1. Import all the required Python Libraries
import pandas as pd
import numpy as np

# ### 2. Locate an Open Source Dataset from the Web
# **Dataset Name**: Titanic - Machine Learning from Disaster
# 
# **Source URL**: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)
# 
# **Data URL used for loading**: `https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv`
# 
# **Description**:
# The Titanic dataset is a standard introductory dataset used for learning data preprocessing and machine learning. It contains demographics and passenger information from the RMS Titanic and whether they survived the disaster.
# 
# **Variable Descriptions**:
# - `PassengerId`: Unique ID for each passenger (Integer)
# - `Survived`: Survival status (0 = No, 1 = Yes) (Categorical/Integer)
# - `Pclass`: Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd) (Categorical/Integer)
# - `Name`: Full name of the passenger (String)
# - `Sex`: Gender of the passenger (Categorical/String)
# - `Age`: Age in years (Quantitative/Float)
# - `SibSp`: Number of siblings/spouses aboard (Quantitative/Integer)
# - `Parch`: Number of parents/children aboard (Quantitative/Integer)
# - `Ticket`: Ticket number sequence (String)
# - `Fare`: Passenger fare cost (Quantitative/Float)
# - `Cabin`: Cabin number (String)
# - `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) (Categorical/String)


### 3. Load the Dataset into a Pandas DataFrame
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)
df.head()

### 4. Data Preprocessing

print("Dimensions:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nStatistics:")
display(df.describe())

### 5. Data Formatting and Data Normalization

print("Original Data Types:\n", df.dtypes)

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Age'] = df['Age'].astype('int')

print("\nConverted Data Types:\n", df.dtypes)

### 6. Turn categorical variables into quantitative variables

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'].astype(str))

display(df.head())