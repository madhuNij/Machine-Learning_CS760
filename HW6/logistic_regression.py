import numpy as np

def sigmoid(z):
    return (1/ (1+np.exp(-z)))

class Linear_LR():
    def __init__(self, learning_rate=1e-4, batch_size=64, epochs=100):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def gradients_db_dw(self, X, y, y_pred):
        m = X.shape[0]
        dw = (1 / m) * np.dot(X.T, (y_pred - y))
        db = (1 / m) * np.sum((y_pred - y))
        return dw, db

    def log_loss(self, y_true, y_pred):
        loss = -np.mean(y_true * (np.log(y_pred)) - (1 - y_true) * np.log(1 - y_pred))
        return loss

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        m, n = X.shape
        w = np.random.randn(n).reshape(n, 1)
        b = 0
        y = y.reshape(m, 1)
        losses = []
        for epoch in range(self.epochs):
            for i in range((m - 1) // self.batch_size + 1):
                s = i * self.batch_size
                e = s + self.batch_size
                xb = X[s:e]
                yb = y[s:e]
                y_pred = sigmoid(np.dot(xb, w) + b)
                dw, db = self.gradients_db_dw(xb, yb, y_pred)
                w -= self.learning_rate * dw
                b -= self.epochs * db
            l = self.log_loss(y, sigmoid(np.dot(X, w) + b))
            losses.append(l)
        self.w = w
        self.b = b
        return w, b, losses

    def predict(self, X):
        prediction = sigmoid(np.dot(X, self.w) + self.b)
        predicted_class = [1 if i > 0.5 else 0 for i in prediction]
        return np.array(predicted_class)