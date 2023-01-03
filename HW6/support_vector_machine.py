import cvxopt
import cvxopt.solvers
import numpy as np


def linear(x1, x2):
    return np.dot(x1, x2)

def rbf_kernel(x, y, sigma=5.0):
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

class SVM():
    def __init__(self, kernel='linear', C=None, degree=2, sigma=5):
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.sigma = sigma
        if self.C is not None:
            self.C = float(self.C)

    def fit(self, X, y):
        m, n = X.shape
        K = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                if self.kernel == 'linear':
                    K[i, j] = linear(X[i], X[j])
                if self.kernel == 'rbf':
                    K[i, j] = rbf_kernel(X[i], X[j], self.sigma)
                    self.C = None
                if self.kernel == 'polynomial':
                    K[i, j] = polynomial_kernel(X[i], X[j], self.degree)

        K = K + (1e-4) * np.eye(m)
        y = y * 1.
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones((m, 1)))
        A = cvxopt.matrix(y, (1, m))
        b = cvxopt.matrix(np.zeros(1))

        if self.C is None or self.C == 0:
            G = cvxopt.matrix(-np.eye(m))
            h = cvxopt.matrix(np.zeros(m))
        else:
            tmp1 = -np.eye(m)
            tmp2 = np.eye(m)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(m)
            tmp2 = np.ones(m) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))


        cvxopt.solvers.options['show_progress'] = False
        cvxopt.solvers.options['abstol'] = 1e-7
        cvxopt.solvers.options['reltol'] = 1e-6
        cvxopt.solvers.options['feastol'] = 1e-7

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(solution['x'])
        sv = alphas > 1e-5
        ind = np.arange(len(alphas))[sv]
        self.alphas = alphas[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]

        self.b = 0
        for p in range(len(self.alphas)):
            # For all support vectors:
            self.b += self.sv_y[p]
            self.b -= np.sum(self.alphas * self.sv_y * K[ind[p], sv])
        self.b = self.b / len(self.alphas)

        if self.kernel == 'linear':
            self.w = np.zeros(n)
            for q in range(len(self.alphas)):
                self.w += self.alphas[q] * self.sv_y[q] * self.sv[q]
        else:
            self.w = None

    def project_points(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.alphas, self.sv_y, self.sv):
                    if self.kernel == 'linear':
                        s += a * sv_y * linear(X[i], sv)
                    if self.kernel == 'rbf':
                        s += a * sv_y * rbf_kernel(X[i], sv, self.sigma)
                    if self.kernel == 'polynomial':
                        s += a * sv_y * polynomial_kernel(X[i], sv, self.degree)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project_points(X))