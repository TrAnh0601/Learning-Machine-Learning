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
    def __init__(self, mode=None, C=1.0, tol=1e-4, fit_intercept=True):
        self.mode = mode
        self.C = C
        self.tol = tol
        self.fit_intercept = fit_intercept

        self.w = None
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = 0

    def fit(self, X, y, learning_rate=0.01, n_iters=1000):
        y = np.asarray(y).reshape(-1, 1)
        Xb = add_bias(X) if self.fit_intercept else X
        m, n = Xb.shape

        self.w = np.zeros((n, 1), dtype=float)

        # 'temperature' (T) controls the "sharpness" of the Sigmoid activation
        # Standard Logistic Regression uses T = 1.0
        # By setting T -> 0, the Sigmoid approximates the Heaviside Step Function,
        # effectively transforming the model into a Perceptron with hard decision boundaries
        temperature = 0.0001 if self.mode == 'perceptron' else 1.0

        # L2 regularization (lambda = 1 / C)
        reg_mask = np.ones_like(self.w)
        if self.fit_intercept:
            reg_mask[0] = 0.0

        for i in range(n_iters):
            z = (Xb @ self.w) / temperature
            y_hat = sigmoid(z)

            # Gradient of log-loss + L2 penalty
            gradient = (Xb.T @ (y - y_hat)) / m - (1 / self.C) * reg_mask * self.w

            # Convergence check
            if np.linalg.norm(gradient) < self.tol:
                self.n_iter_ = i
                break

            self.w += learning_rate * gradient

        else:
            self.n_iter_ = n_iters

        if self.fit_intercept:
            self.coef_ = self.w[1:].ravel()
            self.intercept_ = self.w[0, 0]
        else:
            self.coef_ = self.w.ravel()
            self.intercept_ = 0.0

        return self

    def predict_proba(self, X):
        Xb = add_bias(X) if self.fit_intercept else X
        return sigmoid(Xb @ self.w)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int).ravel()

    def score(self, X, y):
        return (self.predict(X) == y).mean()


if __name__ == "__main__":
    # Test on synthetic binary classification data
    np.random.seed(42)
    X = np.random.randn(1000, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)

    # Standard Logistic Regression
    lr = LogisticRegression()
    lr.fit(X, y, learning_rate=0.1, n_iters=1000)
    acc = (lr.predict(X) == y).mean()
    print(f"Logistic Regression Accuracy: {acc:.2%}")
    print(f"Weights: {lr.coef_}, Bias: {lr.intercept_:.4f}")

    # Perceptron Mode
    perceptron = LogisticRegression(mode='perceptron')
    perceptron.fit(X, y, learning_rate=0.1, n_iters=1000)
    acc_p = (perceptron.predict(X) == y).mean()
    print(f"\nPerceptron Mode Accuracy: {acc_p:.2%}")
    print(f"Weights: {perceptron.coef_}, Bias: {perceptron.intercept_:.4f}")