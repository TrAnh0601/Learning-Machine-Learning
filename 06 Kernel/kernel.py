import numpy as np


class Kernel:
    @staticmethod
    def linear(X1, X2):
        return X1 @ X2.T

    @staticmethod
    def rbf(X1, X2, gamma):
        sq_dist = (np.sum(X1 ** 2, axis=1).reshape(-1, 1) +
                  np.sum(X2 ** 2, axis=1) - 2 * X1 @ X2.T)
        return np.exp(-gamma * sq_dist)

    @staticmethod
    def poly(X1, X2, degree=3, coef=1):
        return (X1 @ X2.T + coef) ** degree