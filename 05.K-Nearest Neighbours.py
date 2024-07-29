import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create a sample DataFrame
data = {
    'Feature1': [2, 4, 4, 4, 6, 6, 6, 8, 8, 8],
    'Feature2': [4, 2, 4, 6, 4, 6, 8, 6, 8, 10],
    'Label':    ['A', 'A', 'A', 'B', 'A', 'B', 'B', 'B', 'B', 'B']
}
df = pd.DataFrame(data)

# Define the KNN function
def knn_predict(df, query, k):
    # Step 1: Calculate distances
    distances = []
    for index, row in df.iterrows():
        distance = np.sqrt((row['Feature1'] - query[0]) ** 2 + (row['Feature2'] - query[1]) ** 2)
        distances.append((distance, row['Label']))
    
    # Step 2: Sort distances and select k-nearest neighbors
    sorted_distances = sorted(distances, key=lambda x: x[0])
    k_nearest_neighbors = sorted_distances[:k]
    
    # Step 3: Perform a majority vote
    k_nearest_labels = [label for _, label in k_nearest_neighbors]
    majority_vote = Counter(k_nearest_labels).most_common(1)[0][0]
    
    return majority_vote

# Example usage of the KNN function
query_point = [5, 5]
k = 3
prediction = knn_predict(df, query_point, k)
print(f'Predicted class for query point {query_point}: {prediction}')

# Evaluate the KNN on the sample data (using train-test split for demonstration)
X = df[['Feature1', 'Feature2']].values
y = df['Label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Make predictions on the test set
predictions = []
for point in X_test:
    predictions.append(knn_predict(pd.DataFrame({'Feature1': X_train[:,0], 'Feature2': X_train[:,1], 'Label': y_train}), point, k))

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')
