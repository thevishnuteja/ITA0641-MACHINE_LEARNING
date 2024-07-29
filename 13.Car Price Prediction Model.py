import pandas as pd

# Load the dataset
data = pd.read_csv(r"C:\Users\sanna\Documents\SIMATS\Courses\Practicals\Machine Learning Lab\Datasets\CarPrice.csv")

# Display the first few rows of the dataset
print(data.head())

# Drop unnecessary columns
data = data.drop(['car_ID', 'CarName'], axis=1)

# Convert categorical variables to dummy variables
data = pd.get_dummies(data, drop_first=True)

# Check for missing values
print(data.isnull().sum())

from sklearn.model_selection import train_test_split

# Define features and target variable
X = data.drop('price', axis=1)
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error, r2_score

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Accuracy: {r2}')
