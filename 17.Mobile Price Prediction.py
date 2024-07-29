import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
# Load the dataset
df = pd.read_csv(r"C:\Users\sanna\Documents\SIMATS\Courses\Practicals\Machine Learning Lab\Datasets\mobile_prices.csv")  # Replace with your dataset path

# Data preprocessing
df = df.dropna()
X = df.drop('price_range', axis=1)  # Features
y = df['price_range']  # Target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print classification report
print(classification_report(y_test, y_pred))

