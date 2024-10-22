import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load and preprocess your dataset
data = pd.read_csv('food_delivery_dataset.csv')  # Replace with your actual dataset path

# Fill missing delivery time values
data['Delivery Time'].fillna(data['Delivery Time'].median(), inplace=True)

# Create categorical delivery time variable
bins = [0, 30, 60, 90, np.inf]  # Define bins (0-30, 31-60, 61-90, 90+ minutes)
labels = ['0-30 mins', '31-60 mins', '61-90 mins', '90+ mins']
data['Delivery Time Category'] = pd.cut(data['Delivery Time'], bins=bins, labels=labels)

# Prepare features and target
X = data.drop(['Delivery Time', 'Delivery Time Category', 'Restaurant'], axis=1)
y = data['Delivery Time Category']

# Convert categorical features to numeric using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Naive Bayes
print("Naive Bayes Classifier:")
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train, y_train)
y_pred_nb = naive_bayes_model.predict(X_test)

# Confusion matrix for Naive Bayes
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
accuracy_nb = accuracy_score(y_test, y_pred_nb)

print("Accuracy (Naive Bayes):", accuracy_nb)
print("Confusion Matrix (Naive Bayes):\n", conf_matrix_nb)

# Plot confusion matrix for Naive Bayes
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_nb, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix - Naive Bayes')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# 2. Decision Tree Classifier
print("\nDecision Tree Classifier:")
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)
y_pred_dt = decision_tree_model.predict(X_test)

# Confusion matrix for Decision Tree
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

print("Accuracy (Decision Tree):", accuracy_dt)
print("Confusion Matrix (Decision Tree):\n", conf_matrix_dt)

# Plot confusion matrix for Decision Tree
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix - Decision Tree')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification report for both classifiers
print("\nClassification Report (Naive Bayes):\n", classification_report(y_test, y_pred_nb))
print("Classification Report (Decision Tree):\n", classification_report(y_test, y_pred_dt))
