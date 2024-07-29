import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
data = {
    'Occupation': ['Engineer', 'Doctor', 'Artist', 'Engineer', 'Artist', 'Doctor', 'Engineer', 'Artist', 'Doctor', 'Engineer'],
    'Credit Score': [720, 680, 650, 700, 710, 690, 730, 640, 660, 750]
}
df = pd.DataFrame(data)

# a. Print the first five rows
print("First five rows of the dataset:")
print(df.head())

# b. Basic statistical computations
print("\nBasic statistical computations:")
print(df.describe())

# c. The columns and their data types
print("\nColumns and their data types:")
print(df.dtypes)

# d. Detect and handle null values
print("\nNull values in the dataset:")
print(df.isnull().sum())

# As an example, let's manually insert a null value and then handle it
df.at[2, 'Credit Score'] = None
print("\nNull values after insertion:")
print(df.isnull().sum())

# Replace null values with the mode
mode_value = df['Credit Score'].mode()[0]
df['Credit Score'] = df['Credit Score'].fillna(mode_value)
print("\nDataset after handling null values:")
print(df)

# e. Explore the dataset using a box plot
sns.boxplot(x='Occupation', y='Credit Score', data=df)
plt.title('Credit Scores Based on Occupation')
plt.show()

# f. Split the dataset into train and test sets
X = pd.get_dummies(df['Occupation'], drop_first=True)  # One-hot encoding for categorical variable
y = df['Credit Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# g. Fit the Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# i. Predict the model
y_pred = lin_reg.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nPredicted values for the test set:")
print(y_pred)
print(f"Model MSE: {mse}")
print(f"Model R2: {r2}")
