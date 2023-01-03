import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt


class CNN():
    def _init_(self, sizes, epochs=30, learning_rate=0.01):
        self.sizes = sizes
        self.epochs = epochs
        self.l_rate = learning_rate
        self.params = self.initialization()

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)
        return 1 / (1 + np.exp(-x))

    def softmax(self, x, derivative=False):
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def initialization(self):
        input_layer = self.sizes[0]
        hidden_1 = self.sizes[1]
        output_layer = self.sizes[2]
        params = {
            'w1': np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
            'w2': np.random.randn(output_layer, hidden_1) * np.sqrt(1. / output_layer)
        }
        return params


    def backward_propagation(self, y_train, output):
        params = self.params
        change_w = {}
        error = 2 * (output - y_train) / output.shape[0] * self.softmax(params['z2'], derivative=True)
        change_w['w2'] = np.outer(error, params['a1'])
        error = np.dot(params['w2'].T, error) * self.sigmoid(params['z1'], derivative=True)
        change_w['w1'] = np.outer(error, params['a0'])
        return change_w

    def forward_pass(self, x_train):
        params = self.params
        params['a0'] = x_train
        params['z1'] = np.dot(params["w1"], params['a0'])
        params['a1'] = self.sigmoid(params['z1'])
        params['z2'] = np.dot(params["w2"], params['a1'])
        params['a2'] = self.softmax(params['z2'])
        return params['a2']

    def update_network_params(self, changes_to_w):
        for key, value in changes_to_w.items():
            self.params[key] -= self.l_rate * value

    def calculate_accuracy(self, x_val, y_val):
        predictions = []
        for x, y in zip(x_val, y_val):
            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))
        return np.mean(predictions)

    def train_model(self, x_train, y_train, x_val, y_val):
        accuracy = []
        for iteration in range(self.epochs):
            for x, y in zip(x_train, y_train):
                output = self.forward_pass(x)
                changes_to_w = self.backward_propagation(y, output)
                self.update_network_params(changes_to_w)

            accuracy = self.calculate_accuracy(x_val, y_val)
            accuracy.append(accuracy)
            print('Epoch: {0}, Accuracy: {2:.2f}%'.format(
                iteration + 1, accuracy * 100
            ))
        test_error = []
        for i in accuracy:
            test_error.append(1 - (accuracy/100))
        epoch = np.arange(0, self.epochs)
        plt.plot(epoch, test_error, label='Test Error')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.legend(loc="upper right")
        plt.show()


data, label = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
data = (data / 255).astype('float32')
label = to_categorical(label)

x_train, x_val, y_train, y_val = train_test_split(data, label, test_size=0.15, random_state=42)
dnn = CNN(sizes=[784, 300, 10])
dnn.train_model(x_train, y_train, x_val, y_val)