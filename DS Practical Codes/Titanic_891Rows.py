# Generated from: Titanic_891Rows.ipynb
# Converted at: 2026-04-23T03:19:08.818Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# Title: Data Visualization I


# PROBLEM STATEMENT: 
# 1. Use the inbuilt dataset 'titanic'. The dataset contains 891 rows and contains information about the 
# passengers who boarded the unfortunate Titanic ship. Use the Seaborn library to see if we can find any 
# patterns in the data. 
# 2. Write a code to check how the price of the ticket (column name: 'fare') for each passenger is distributed by 
# plotting a histogram. 


import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')

# View first 5 rows
titanic.head()

print(titanic.info())
print(titanic.describe())

sns.countplot(x='survived', hue='sex', data=titanic)
plt.title("Survival based on Gender")
plt.show()

sns.countplot(x='survived', hue='class', data=titanic)
plt.title("Survival based on Class")
plt.show()

sns.histplot(titanic['age'].dropna(), bins=30)
plt.title("Age Distribution")
plt.show()

sns.histplot(titanic['fare'], bins=30)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.show()

sns.histplot(titanic['fare'], bins=50)
plt.title("Fare Distribution (More Bins)")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.show()

# Most passengers paid low fare, few paid very high → skewed distribution


#