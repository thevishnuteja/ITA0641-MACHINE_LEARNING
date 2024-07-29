import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Data Preparation
# Creating a sample dataframe
data = {
    'feature1': [2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
    'feature2': [1, 1, 2, 3, 5, 8, 13, 21, 34, 55],
    'label': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

# Splitting data into training and testing sets
X = df[['feature1', 'feature2']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 2: Model Initialization
model = LogisticRegression()

# Step 3: Model Training
model.fit(X_train, y_train)

# Step 4: Prediction
y_pred = model.predict(X_test)

# Step 5: Performance Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Output
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
