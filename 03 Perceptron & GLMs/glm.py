import numpy as np


def add_bias(X):
    ones = np.ones((X.shape[0], 1))
    return np.concatenate((ones, X), axis=1)


class Gaussian:
    @staticmethod
    def link_inv(eta):
        return eta

class Bernoulli:
    @staticmethod
    def link_inv(eta):
        return 1 / (1 + np.exp(-eta))

class Poisson:
    @ staticmethod
    def link_inv(eta):
        return np.exp(eta)


class GLM:
    def __init__(self, distribution):
        self.dist = distribution
        self.w = None

    def fit(self, X, y, learning_rate=0.01, n_iters=1000):
        Xb = add_bias(X)
        m, n = Xb.shape
        y = y.reshape(-1, 1) if y.ndim == 1 else y

        self.w = np.zeros((n, 1), dtype=float)

        for _ in range(n_iters):
            eta = Xb @ self.w
            y_hat = self.dist.link_inv(eta)

            gradient = (Xb.T @ (y - y_hat)) / m
            self.w += learning_rate * gradient

    def predict(self, X):
        Xb = add_bias(X)
        eta = Xb @ self.w

        predict = self.dist.link_inv(eta)

        if self.dist == Bernoulli:
            return (predict >= 0.5).astype(int)

        return predict


if __name__ == "__main__":
    np.random.seed(42)

    # Logistic Regression test
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)

    logistic = GLM(distribution=Bernoulli)
    logistic.fit(X, y)

    acc = (logistic.predict(X) == y).mean()
    print(f"\nLogistic Regression Model Accuracy: {acc:.2%}")

    # Linear Regression Test
    X = np.random.randn(10, 1)
    y = 3 + 2 * X.ravel() + 0.5 * np.random.randn(10)

    linear = GLM(distribution=Gaussian)
    linear.fit(X, y)

    print(f"LR Weight: {linear.w[1:]}, {linear.w[0, 0]}")