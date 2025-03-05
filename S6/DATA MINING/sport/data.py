# Our Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Our Player Data
data = pd.read_csv(r'C:\Users\Othman SALAHI\Desktop\STUDY\DATA MINING\sport\injury_data.csv')
print(data)

# Exploratory Data Analysis
# data.info()
# data.describe()

# Countplot for injury likelihood
# sns.countplot(x='Likelihood_of_Injury', data=data)

# Correlation plot
# data.corr()['Likelihood_of_Injury'].sort_values().plot(kind='bar')

# Plot Training Intensity vs Likelihood of Injury
# plt.figure(figsize=(20,8))
# sns.displot(x='Training_Intensity', data=data, hue='Likelihood_of_Injury')

# # Countplot for Previous Injuries vs Likelihood of Injury
# plt.figure(figsize=(10,8))
# sns.countplot(x='Previous_Injuries', data=data, hue='Likelihood_of_Injury', palette='viridis')

# # Plot Player Height distribution
# sns.displot(x='Player_Height', data=data)

# # Plot Player Age vs Likelihood of Injury
# sns.displot(x='Player_Age', data=data, hue='Likelihood_of_Injury')

# Data PreProcessing
X = data.drop(['Likelihood_of_Injury', 'Recovery_Time'], axis=1)
y = data['Likelihood_of_Injury']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Our model
model = LogisticRegression()
model.fit(X_train, y_train)

# Model evaluation
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Model accuracy
print(f'Our model training-prediction accuracy is {accuracy_score(train_pred, y_train) * 100:.2f}%')
print(f'Our model test-prediction accuracy is {accuracy_score(test_pred, y_test) * 100:.2f}%')

# Metrics
print(classification_report(test_pred, y_test))
print(confusion_matrix(test_pred, y_test))
