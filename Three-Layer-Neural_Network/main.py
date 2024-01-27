import numpy as np
import pandas as pd


class MLP:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size

        self.weights_input_hidden1 = np.random.randn(self.input_size, self.hidden_size1)
        self.bias_input_hidden1 = np.random.randn(1, self.hidden_size1)

        self.weights_hidden1_hidden2 = np.random.randn(self.hidden_size1, self.hidden_size2)
        self.bias_hidden1_hidden2 = np.random.randn(1, self.hidden_size2)

        self.weights_hidden2_output = np.random.randn(self.hidden_size2, self.output_size)
        self.bias_hidden2_output = np.random.randn(1, self.output_size)

        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def _softmax(self, x):
        exp_vals = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    def _softmax_derivative(self, x):
        s = self._softmax(x)
        return s * (1 - s)

    def fit(self, X, y, epochs=1000):
        for epoch in range(epochs):

            hidden_layer1_input = np.dot(X, self.weights_input_hidden1) + self.bias_input_hidden1
            hidden_layer1_output = self.sigmoid(hidden_layer1_input)

            hidden_layer2_input = np.dot(hidden_layer1_output, self.weights_hidden1_hidden2) + self.bias_hidden1_hidden2
            hidden_layer2_output = self.sigmoid(hidden_layer2_input)

            output_layer_input = np.dot(hidden_layer2_output, self.weights_hidden2_output) + self.bias_hidden2_output
            output_layer_output = self.sigmoid(output_layer_input)

            error = y - output_layer_output
            d_output = error * self.sigmoid_derivative(output_layer_output)

            error_hidden2_layer = d_output.dot(self.weights_hidden2_output.T)
            d_hidden2_layer = error_hidden2_layer * self.sigmoid_derivative(hidden_layer2_output)

            error_hidden1_layer = d_hidden2_layer.dot(self.weights_hidden1_hidden2.T)
            d_hidden1_layer = error_hidden1_layer * self.sigmoid_derivative(hidden_layer1_output)

            self.weights_hidden2_output += hidden_layer2_output.T.dot(d_output) * self.learning_rate
            self.bias_hidden2_output += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate

            self.weights_hidden1_hidden2 += hidden_layer1_output.T.dot(d_hidden2_layer) * self.learning_rate
            self.bias_hidden1_hidden2 += np.sum(d_hidden2_layer, axis=0, keepdims=True) * self.learning_rate

            self.weights_input_hidden1 += X.T.dot(d_hidden1_layer) * self.learning_rate
            self.bias_input_hidden1 += np.sum(d_hidden1_layer, axis=0, keepdims=True) * self.learning_rate

    def predict(self, X):
        hidden_layer1_input = np.dot(X, self.weights_input_hidden1) + self.bias_input_hidden1
        hidden_layer1_output = self.sigmoid(hidden_layer1_input)

        hidden_layer2_input = np.dot(hidden_layer1_output, self.weights_hidden1_hidden2) + self.bias_hidden1_hidden2
        hidden_layer2_output = self.sigmoid(hidden_layer2_input)

        output_layer_input = np.dot(hidden_layer2_output, self.weights_hidden2_output) + self.bias_hidden2_output
        output_layer_output = self.sigmoid(output_layer_input)

        return output_layer_output


def train_test_split(data, labels, test_size=0.2):

    train_size = int(len(data) * (1 - test_size))
    X_train = data[:train_size]
    X_test = data[train_size:]
    y_train = labels[:train_size]
    y_test = labels[train_size:]

    return X_train, X_test, y_train, y_test


def evaluate_model(y_true, y_pred):
    accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

    true_positives = np.sum(np.argmax(y_pred, axis=1) & np.argmax(y_true, axis=1))
    predicted_positives = np.sum(np.argmax(y_pred, axis=1))
    actual_positives = np.sum(np.argmax(y_true, axis=1))

    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {

        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score

    }


def confusion_matrix(y_true, y_pred, num_classes=3):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(len(y_true)):
        true_label = np.argmax(y_true[i])
        predicted_label = np.argmax(y_pred[i])
        confusion_matrix[true_label][predicted_label] += 1

    print("Confusion Matrix:")
    print("True\Predicted".rjust(15), end='')
    for i in range(len(confusion_matrix)):
        print(str(i).rjust(5), end='')
    print("\n")

    for i, row in enumerate(confusion_matrix):
        print(f"True {i}".rjust(15), end='')
        for value in row:
            print(str(value).rjust(5), end='')
        print()


input_data = pd.read_csv('Inputs.csv').transpose()
output_labels = pd.read_csv('Targets.csv').transpose()


X_train, X_test, y_train, y_test = train_test_split(input_data, output_labels, test_size=0.2)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


input_size = X_train.shape[1]
hidden_layer_units1 = 64
hidden_layer_units2 = 32
output_size = len(set(output_labels))

model = MLP(input_size, hidden_layer_units1, hidden_layer_units2, output_size, learning_rate=0.01)

model.fit(X_train, y_train, epochs=100)

y_pred = model.predict(X_test)


evaluation_result = evaluate_model(y_test, y_pred)
print("Evaluation Results:")
for metric, value in evaluation_result.items():
    print(f"{metric}: {value:.4f}")


confusion_matrix(y_test, y_pred)




