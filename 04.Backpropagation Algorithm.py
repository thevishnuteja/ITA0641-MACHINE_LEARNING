import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Function to train the neural network
def train(X, y, epochs, learning_rate):
    input_layer_neurons = X.shape[1]
    hidden_layer_neurons = 2
    output_neurons = 1

    # Weight and bias initialization
    hidden_weights = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
    hidden_bias = np.random.uniform(size=(1, hidden_layer_neurons))
    output_weights = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
    output_bias = np.random.uniform(size=(1, output_neurons))

    for _ in range(epochs):
        # Forward Propagation
        hidden_layer_activation = np.dot(X, hidden_weights)
        hidden_layer_activation += hidden_bias
        hidden_layer_output = sigmoid(hidden_layer_activation)

        output_layer_activation = np.dot(hidden_layer_output, output_weights)
        output_layer_activation += output_bias
        predicted_output = sigmoid(output_layer_activation)

        # Backpropagation
        error = y - predicted_output
        d_predicted_output = error * sigmoid_derivative(predicted_output)

        error_hidden_layer = d_predicted_output.dot(output_weights.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

        # Updating Weights and Biases
        output_weights += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
        output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
        hidden_weights += X.T.dot(d_hidden_layer) * learning_rate
        hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    return hidden_weights, hidden_bias, output_weights, output_bias

# Function to predict new data
def predict(X, hidden_weights, hidden_bias, output_weights, output_bias):
    hidden_layer_activation = np.dot(X, hidden_weights)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output, output_weights)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)
    return predicted_output

# Example dataset (XOR gate)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Training the neural network
epochs = 10000
learning_rate = 0.1
hidden_weights, hidden_bias, output_weights, output_bias = train(X, y, epochs, learning_rate)

# Testing the neural network
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predicted_output = predict(test_data, hidden_weights, hidden_bias, output_weights, output_bias)

print("Predicted Output:")
print(predicted_output)
