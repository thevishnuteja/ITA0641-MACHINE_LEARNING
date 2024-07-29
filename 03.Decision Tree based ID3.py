import pandas as pd
import numpy as np

# Example dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Function to calculate entropy
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    return -np.sum([(counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])

# Function to calculate information gain
def info_gain(data, split_attribute, target_attribute="PlayTennis"):
    total_entropy = entropy(data[target_attribute])
    vals, counts = np.unique(data[split_attribute], return_counts=True)
    weighted_entropy = np.sum([(counts[i]/np.sum(counts)) * entropy(data[data[split_attribute] == vals[i]][target_attribute]) for i in range(len(vals))])
    return total_entropy - weighted_entropy

# ID3 Algorithm
def ID3(data, original_data, features, target_attribute="PlayTennis"):
    if len(np.unique(data[target_attribute])) == 1:
        return np.unique(data[target_attribute])[0]
    elif len(data) == 0:
        return np.unique(original_data[target_attribute])[np.argmax(np.unique(original_data[target_attribute], return_counts=True)[1])]
    elif len(features) == 0:
        return np.unique(data[target_attribute])[np.argmax(np.unique(data[target_attribute], return_counts=True)[1])]
    else:
        parent_node = np.unique(data[target_attribute])[np.argmax(np.unique(data[target_attribute], return_counts=True)[1])]
        item_values = [info_gain(data, feature) for feature in features]
        best_feature = features[np.argmax(item_values)]
        tree = {best_feature: {}}
        features = [i for i in features if i != best_feature]
        for value in np.unique(data[best_feature]):
            sub_data = data[data[best_feature] == value]
            subtree = ID3(sub_data, original_data, features, target_attribute)
            tree[best_feature][value] = subtree
        return tree

# Build the decision tree
tree = ID3(df, df, df.columns[:-1])
print(tree)
print()

# Function to classify a new example
def classify(example, tree):
    attribute = list(tree.keys())[0]
    if example[attribute] in tree[attribute]:
        result = tree[attribute][example[attribute]]
        if isinstance(result, dict):
            return classify(example, result)
        else:
            return result
    else:
        return "No"

# New sample to classify
example = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Strong'}
print("Classified as:", classify(example, tree))
