import numpy as np
from kernel import Kernel


class SVM:
    def __init__(self, kernel='linear', C=1.0, gamma=1.0, degree=3, tol=1e-3):
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.tol = tol

        if kernel == 'linear':
            self.kernel = Kernel.linear
        elif kernel == 'rbf':
            self.kernel = lambda X1, X2: Kernel.rbf(X1, X2, gamma)
        elif kernel == 'poly':
            self.kernel = lambda X1, X2: Kernel.poly(X1, X2, degree)

    def fit(self, X, y, n_iters=1000):
        n_samples = X.shape[0]
        self.X = X

        y = np.where(y <= 0, -1, 1)
        self.y = y

        self.alpha = np.zeros(n_samples)
        self.b = 0

        K = self.kernel(X, X)

        # SMO algorithm
        for _ in range(n_iters):
            alpha_prev = np.copy(self.alpha)

            for i in range(n_samples):
                E_i = self._decision_func(X[i:i+1])[0] - y[i]

                if (y[i] * E_i < -self.tol and self.alpha[i] < self.C) or \
                   (y[i] * E_i > self.tol and self.alpha[i] > 0):

                    j = np.random.choice([x for x in range(n_samples) if x != i])
                    E_j = self._decision_func(X[j:j+1])[0] - y[j]

                    alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]

                    # Compute bounds
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])

                    if L == H:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    # Update alpha_j
                    self.alpha[j] -= y[j] * (E_i - E_j) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)

                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    # Update alpha_i
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])

                    # Update bias
                    b1 = (self.b - E_i - y[i] * (self.alpha[i] - alpha_i_old) * K[i, i]
                        - y[j] * (self.alpha[j] - alpha_j_old) * K[i, j])
                    b2 = (self.b - E_j - y[i] * (self.alpha[i] - alpha_i_old) * K[i, j]
                        - y[j] * (self.alpha[j] - alpha_j_old) * K[j, j])

                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

            if np.linalg.norm(self.alpha - alpha_prev) < self.tol:
                break

        # Suport vectors
        sv = self.alpha > 1e-5
        self.support_vectors = X[sv]
        self.support_labels = y[sv]
        self.support_alphas = self.alpha[sv]

    def _decision_func(self, X):
        K = self.kernel(X, self.X)
        return K @ (self.alpha * self.y) + self.b

    def predict(self, X):
        return np.sign(self._decision_func(X))


if __name__ == '__main__':
    import numpy as np
    from svm import SVM


    def run_test(X_test, y_test, kernel='linear'):
        model = SVM(kernel=kernel)
        model.fit(X_test, y_test, n_iters=1000)
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
        np.array([0]*20 + [1]*20),
    )

    # 2. Overlapping Data
    np.random.seed(42)
    X_random = np.vstack([np.random.randn(20, 2) + 2, np.random.randn(20, 2) + 6])
    y_random = np.array([0] * 20 + [1] * 20)
    run_test(X_random, y_random)

    # 3. Outliers
    X_edge = np.array([[1, 1], [2, 2], [2, 1], [2.1, 2.1], [3, 3]])
    y_edge = np.array([0, 0, 0, 1, 1])
    run_test(X_edge, y_edge, kernel='rbf')

    # 4. Extreme Outliers
    X_outlier_hard = np.array([
        [1, 2], [2, 1], [3, 2], [2, 3],
        [8, 7], [7, 8], [9, 8], [8, 9],
        [8, 8], [7, 7],
        [2, 2], [1, 1]
    ])
    y_outlier_hard = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1])
    run_test(X_outlier_hard, y_outlier_hard, kernel='poly')

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
    run_test(X_xor, y_xor, kernel='rbf')