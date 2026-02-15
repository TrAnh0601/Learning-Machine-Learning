import numpy as np


class SVM:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y, learning_rate=0.01, lambda_param=0.01, n_iters=1000):
        y = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0

        for _ in range(n_iters):
            for idx, x in enumerate(X):
                condition = y[idx] * (np.dot(x, self.w) + self.b) >= 1

                if condition:
                    self.w -= learning_rate * (2 * lambda_param * self.w)
                else:
                    self.w -= learning_rate * (2 * lambda_param * self.w - x * y[idx])
                    self.b += learning_rate * y[idx]

        return self

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)


if __name__ == "__main__":
    def run_test(X_test, y_test):
        model = SVM()
        model.fit(X_test, y_test, learning_rate=0.01, lambda_param=0.01, n_iters=1000)
        preds = model.predict(X_test).flatten()
        actual = np.where(y_test <= 0, -1, 1)
        acc = np.mean(preds == actual)
        print(f"Accuracy: {acc * 100:.2f}%")

    # 1. Simple
    run_test(
        np.concatenate([
            np.random.randn(20, 2) + [2, 2],
            np.random.randn(20, 2) + [8, 8]
        ]),
            np.array([0]*20 + [1]*20)
    )

    # 2. Overlapping Data
    np.random.seed(42)
    X_random = np.vstack([np.random.randn(20, 2) + 2, np.random.randn(20, 2) + 6])
    y_random = np.array([0] * 20 + [1] * 20)
    run_test(X_random, y_random)

    # 3. Outliers
    X_edge = np.array([[1, 1], [2, 2], [2, 1], [2.1, 2.1], [3, 3]])
    y_edge = np.array([0, 0, 0, 1, 1])
    run_test(X_edge, y_edge)

    # 4. Extreme Outliers
    X_outlier_hard = np.array([
        [1, 2], [2, 1], [3, 2], [2, 3],
        [8, 7], [7, 8], [9, 8], [8, 9],
        [8, 8], [7, 7],
        [2, 2], [1, 1]
    ])
    y_outlier_hard = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1])
    run_test(X_outlier_hard, y_outlier_hard)

    # 5. Heavy Overlap
    np.random.seed(42)
    X_heavy = np.vstack([
        np.random.randn(50, 2) + 4,
        np.random.randn(50, 2) + 5
    ])
    y_heavy = np.array([0] * 50 + [1] * 50)
    run_test(X_heavy, y_heavy)

    # 6. XOR
    X_xor = np.array([
        [1, 1], [2, 2],
        [8, 8], [9, 9],
        [1, 9], [2, 8],
        [9, 1], [8, 2]
    ])
    y_xor = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    run_test(X_xor, y_xor)