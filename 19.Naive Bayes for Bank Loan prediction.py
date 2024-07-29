import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv(r"C:\Users\sanna\Documents\SIMATS\Courses\Practicals\Machine Learning Lab\Datasets\train_loan.csv")

# Fill missing values
data = data.ffill()

# Encode categorical variables
le = LabelEncoder()
for column in data.select_dtypes(include=['object']).columns:
    data[column] = le.fit_transform(data[column])

# Check for any remaining NaN values
if data.isnull().values.any():
    print("Warning: There are still NaN values in the dataset.")
    data = data.dropna()  # Alternatively, you can use other imputation methods

# Split the data into features and target variable
X = data.drop('Loan_Status', axis=1)  # Features
y = data['Loan_Status']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Gaussian Naive Bayes model
nb = GaussianNB()
nb.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nb.predict(X_test)

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))
