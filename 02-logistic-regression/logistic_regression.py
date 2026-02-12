import numpy as np

# UTILITY FUNCTIONS
def add_bias(X):
    """Add intercept term (column of ones) to feature matrix"""
    ones = np.ones((X.shape[0], 1))
    return np.concatenate((ones, X), axis=1)

def sigmoid(z):
    """Sigmoid function"""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


class LogisticRegression:
    def __init__(self, mode=None):
        self.w = None
        self.mode = mode

    def fit(self, X, y, learning_rate=0.01, n_iters=1000):
        Xb = add_bias(X)
        m, n = Xb.shape

        self.w = np.zeros((n, 1), dtype=float)
        temperature = 1.0

        if self.mode == 'perceptron':
            temperature = 0.0001

        for _ in range(n_iters):
            z = (Xb @ self.w) / temperature
            y_hat = sigmoid(z)
            gradient = (Xb.T @ (y - y_hat)) / m

            self.w += learning_rate * gradient

        return self

    def predict_proba(self, X):
        Xb = add_bias(X)
        return sigmoid(Xb @ self.w)

    def predict(self, X, threshold=0.5):
        Xb = add_bias(X)
        return (self.predict_proba(X) >= threshold).astype(int)


if __name__ == "__main__":
    # Test on synthetic binary classification data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)

    # Standard Logistic Regression
    lr = LogisticRegression()
    lr.fit(X, y, learning_rate=0.1, n_iters=1000)
    acc = (lr.predict(X) == y).mean()
    print(f"Logistic Regression Accuracy: {acc:.2%}")
    print(f"Weights: {lr.w[1:].ravel()}, Bias: {lr.w[0, 0]:.4f}")

    # Perceptron Mode (fix the bug first!)
    perceptron = LogisticRegression(mode='perceptron')
    perceptron.fit(X, y, learning_rate=0.1, n_iters=1000)
    acc_p = (perceptron.predict(X) == y).mean()
    print(f"\nPerceptron Mode Accuracy: {acc_p:.2%}")
    print(f"Weights: {perceptron.w[1:].ravel()}, Bias: {perceptron.w[0, 0]:.4f}")

    # Show some predictions
    print(f"\nSample predictions:")
    print(f"True labels: {y[:10].ravel()}")
    print(f"Predicted:   {lr.predict(X[:10]).ravel()}")
    print(f"Probabilities: {lr.predict_proba(X[:10]).ravel()}")