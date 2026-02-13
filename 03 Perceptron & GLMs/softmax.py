import numpy as np


def add_bias(X):
    ones = np.ones((X.shape[0], 1))
    return np.concatenate((ones, X), axis=1)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def one_hot_encode(y, n_classes):
    m = len(y)
    one_hot = np.zeros((m, n_classes))
    one_hot[np.arange(m), y] = 1
    return one_hot


class Softmax:
    def __init__(self):
        self.w = None

    def fit(self, X, y, learning_rate=0.01, n_iters=100):
        Xb = add_bias(X)
        m, n = Xb.shape

        n_classes = len(np.unique(y))
        y = one_hot_encode(y, n_classes)

        self.w = np.zeros((n, n_classes), dtype=float)

        for _ in range(n_iters):
            z = Xb @ self.w
            p_hat = softmax(z)

            gradient = (Xb.T @ (y - p_hat)) / m
            self.w += learning_rate * gradient

        return self

    def predict(self, X):
        Xb = add_bias(X)
        z = Xb @ self.w
        p_hat = softmax(z)
        return np.argmax(p_hat, axis=1)


if __name__ == "__main__":
    np.random.seed(42)

    # Create 3 clusters (3 classes)
    n_samples = 150
    n_features = 2

    # Class 0: centered at (0, 0)
    X0 = np.random.randn(50, 2) + np.array([0, 0])
    y0 = np.zeros(50, dtype=int)

    # Class 1: centered at (3, 3)
    X1 = np.random.randn(50, 2) + np.array([3, 3])
    y1 = np.ones(50, dtype=int)

    # Class 2: centered at (6, 0)
    X2 = np.random.randn(50, 2) + np.array([6, 0])
    y2 = np.full(50, 2, dtype=int)

    # Combine data
    X = np.vstack([X0, X1, X2])
    y = np.hstack([y0, y1, y2])

    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]

    # Train the model
    model = Softmax()
    model.fit(X, y, learning_rate=0.1, n_iters=1000)

    # Make predictions
    predictions = model.predict(X)

    # Calculate accuracy
    accuracy = np.mean(predictions == y)
    print(f"Training Accuracy: {accuracy * 100:.2f}%")