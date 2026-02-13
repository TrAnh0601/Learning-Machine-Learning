import numpy as np


def add_bias(X):
    ones = np.ones((X.shape[0], 1))
    return np.concatenate((ones, X), axis=1)


class Perceptron:
    def __init__(self):
        self.w = None

    def fit(self, X, y, learning_rate=1.0, n_iters=1000):
        Xb = add_bias(X)
        m, n = Xb.shape

        self.w = np.zeros((n, 1), dtype=float)

        for _ in range(n_iters):
            z = Xb @ self.w
            y_hat = (z >= 0).astype(int)
            errors = y - y_hat

            # Stop if no errors (convergence)
            if np.all(errors == 0):
                break

            self.w += learning_rate * (Xb.T @ errors)

        return self

    def predict(self, X):
        Xb = add_bias(X)
        return (Xb @ self.w >= 0).astype(int)


if __name__ == "__main__":
    # Test on synthetic binary classification data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)

    perceptron = Perceptron()
    perceptron.fit(X, y)
    acc = (perceptron.predict(X) == y).mean()

    print(f"\nPerceptron Model Accuracy: {acc:.2%}")