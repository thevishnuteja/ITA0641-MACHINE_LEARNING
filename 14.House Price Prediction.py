import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load the Data
data = pd.read_csv(r"C:\Users\sanna\Documents\SIMATS\Courses\Practicals\Machine Learning Lab\Datasets\HousePricePrediction.csv")

# Step 2: Data Cleaning
# Select only numeric columns
numeric_cols = data.select_dtypes(include=['number']).columns

# Fill missing values in numeric columns with their respective medians
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# Check for remaining missing values
print(data.isnull().sum())

# Step 3: Feature Selection
# Select features and target variable
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']

# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Step 4: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Choose a Machine Learning Algorithm
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Step 6: Train the Model
model.fit(X_train, y_train)

# Step 7: Make Predictions
predictions = model.predict(X_test)

# Step 8: Evaluate the Model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Accuracy: {r2}')
