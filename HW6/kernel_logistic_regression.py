import numpy as np

def sigmoid(z):
    return (1.0 / (1 + np.exp(-z)))

class KLR():
    def __init__(self, learning_rate=1e-4, kernel='polynomial', epochs=100):
        self.learning_rate = learning_rate
        self.kernel = kernel
        self.epochs = epochs

    def kernel(self, x, y, param=5):
        if self.kernel == 'polynomial':
            return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * (param ** 2)))
        else:
            return (1 + np.dot(x, y)) ** param

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        m, n = X.shape

        self.train = X
        self.train_length = m

        self.alphas = np.zeros(m)
        self.bias = 0
        self.gram = np.zeros((m, m))

        for i in range(m):
            for j in range(m):
                self.gram[i, j] = self.kernel(X[i], X[j])

        for epoch in range(self.epochs):
            for i in range(self.train_length):
                x_test = X[i].reshape(1, -1)
                predicted = self.predict(x_test)
                gradient = (predicted - y[i]).item()
                for j in range(self.train_length):
                    self.alphas[j] += self.learning_rate * gradient * self.gram[i, j]
                self.bias += self.learning_rate * gradient

    def predict(self, X):
        m = X.shape[0]
        pred_class = []
        for i in range(m):
            z = 0
            for j in range(self.train_length):
                z += self.alphas[j] * self.kernel(X[i], self.train[j])
            z += self.bias
            pred_class.append(sigmoid(z))
        return np.array(pred_class)