import numpy as np

def sigmoid(z):
    return (1/ (1+np.exp(-z)))


class LR():
    def __init__(self, learning_rate=1e-4, batch_size=64, epochs=100):
        self.lr = learning_rate
        self.bs = batch_size
        self.epochs = epochs

    def normalize(self, X):

        # X --> Input.

        # m-> number of training examples
        # n-> number of features
        m, n = X.shape

        # Normalizing all the n features of X.
        for i in range(n):
            X = (X - X.mean(axis=0)) / X.std(axis=0)

        return X

    def fit(self, X, y):

        # X --> Input.
        # y --> true/target value.
        # bs --> Batch Size.
        # epochs --> Number of iterations.
        # lr --> Learning rate.

        # m-> number of training examples
        # n-> number of features
        m, n = X.shape

        # Initializing weights and bias to zeros.
        w = np.zeros((n, 1))
        b = 0

        # Reshaping y.
        y = y.reshape(m, 1)

        # Normalizing the inputs.
        x = self.normalize(X)

        # Empty list to store losses.
        losses = []

        # Training loop.
        for epoch in range(self.epochs):
            for i in range((m - 1) // self.bs + 1):
                # Defining batches. SGD.
                start_i = i * self.bs
                end_i = start_i + self.bs
                xb = X[start_i:end_i]
                yb = y[start_i:end_i]

                # Calculating hypothesis/prediction.
                y_hat = sigmoid(np.dot(xb, w) + b)

                # Getting the gradients of loss w.r.t parameters.
                dw, db = self.gradients(xb, yb, y_hat)

                # Updating the parameters.
                w -= self.lr * dw
                b -= self.lr * db

            # Calculating loss and appending it in the list.
            l = self.loss(y, sigmoid(np.dot(X, w) + b))
            losses.append(l)

        # returning weights, bias and losses(List).
        return w, b, losses
    def gradients(self, X, y, y_hat):
        # X --> Input.
        # y --> true/target value.
        # y_hat --> hypothesis/predictions.
        # w --> weights (parameter).
        # b --> bias (parameter).

        # m-> number of training examples.
        m = X.shape[0]

        # Gradient of loss w.r.t weights.
        dw = (1 / m) * np.dot(X.T, (y_hat - y))

        # Gradient of loss w.r.t bias.
        db = (1 / m) * np.sum((y_hat - y))

        return dw, db
    def loss(self, y, y_hat):
        loss = -np.mean(y * (np.log(y_hat)) - (1 - y) * np.log(1 - y_hat))
        return loss

    def predict(self,X):

        # X --> Input.

        # Normalizing the inputs.
        x = self.normalize(X)

        # Calculating presictions/y_hat.
        preds = sigmoid(np.dot(X, w) + self.b)

        # Empty List to store predictions.
        pred_class = []
        # if y_hat >= 0.5 --> round up to 1
        # if y_hat < 0.5 --> round up to 1
        pred_class = [1 if i > 0.5 else 0 for i in preds]

        return np.array(pred_class)