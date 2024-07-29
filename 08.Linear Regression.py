import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Create Dataframe
# Generating a sample dataset
np.random.seed(0)
X = np.random.rand(100, 1) * 10  # Independent variable
y = 2.5 * X + np.random.randn(100, 1) * 2  # Dependent variable with some noise

# Convert to DataFrame
df = pd.DataFrame(data={'X': X.flatten(), 'y': y.flatten()})

# Step 3: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(df[['X']], df['y'], test_size=0.2, random_state=0)
# Step 4: Train Model
model = LinearRegression()
model.fit(X_train, y_train)
# Step 5: Evaluate Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# Plotting the results
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()
# Output results
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
