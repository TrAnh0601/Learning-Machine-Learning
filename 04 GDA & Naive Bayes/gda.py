import numpy as np


def multivariate_gaussian(x, mu, sigma):
    d = len(mu)
    sigma_inv = np.linalg.inv(sigma)
    sigma_det = np.linalg.det(sigma)

    denominator = np.sqrt((2 * np.pi) ** d * sigma_det)
    diff = x - mu
    exponent = -0.5 * diff.T @ sigma_inv @ diff

    return (1 / denominator) * np.exp(exponent)


class GDA:
    def __init__(self):
        self.phi = None
        self.mu_0 = None
        self.mu_1 = None
        self.sigma = None

    def fit(self, X, y):
        m, n = X.shape

        # Maximum likelihood estimates
        # φ = (1/m) * Σ 1{y_i = 1}
        self.phi = np.mean(y)

        # μ_0 = (Σ 1{y_i = 0} * x_i) / (Σ 1{y_i = 0})
        self.mu_0 = X[y == 0].mean(axis=0)

        # μ_1 = (Σ 1{y_i = 1} * x_i) / (Σ 1{y_i = 1})
        self.mu_1 = X[y == 1].mean(axis=0)

        # Σ = (1/m) * Σ (x_i - μ_{y_i})(x_i - μ_{y_i})^T
        self.sigma = np.zeros((n, n))
        for i in range(m):
            if y[i] == 0:
                diff = (X[i] - self.mu_0).reshape(-1, 1)
            else:
                diff = (X[i] - self.mu_1).reshape(-1, 1)
            self.sigma += diff @ diff.T
        self.sigma /= m

    def predict_proba(self, X):
        probabilities = []

        for x in X:
            # P(x|y=0) * P(y=0)
            p_x_given_0 = multivariate_gaussian(x, self.mu_0, self.sigma)
            p_0 = 1 - self.phi

            # P(x|y=1) * P(y=1)
            p_x_given_1 = multivariate_gaussian(x, self.mu_1, self.sigma)
            p_1 = self.phi

            # P(y=1|x) using Bayes rule
            numerator = p_x_given_1 * p_1
            denominator = p_x_given_0 * p_0 + p_x_given_1 * p_1

            prob_y1 = numerator / denominator
            probabilities.append(prob_y1)

        return np.array(probabilities)

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


def generate_synthetic_data(n_samples=100, n_features=2, seed=42):
    """Generate synthetic binary classification data"""
    np.random.seed(seed)

    # Class 0: centered at [-2, -2]
    X_0 = np.random.randn(n_samples // 2, n_features) + np.array([-2, -2])
    y_0 = np.zeros(n_samples // 2)

    # Class 1: centered at [2, 2]
    X_1 = np.random.randn(n_samples // 2, n_features) + np.array([2, 2])
    y_1 = np.ones(n_samples // 2)

    # Combine
    X = np.vstack([X_0, X_1])
    y = np.hstack([y_0, y_1])

    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]

    return X, y


if __name__ == "__main__":
    # Generate synthetic data
    X_train, y_train = generate_synthetic_data(n_samples=200, seed=42)
    X_test, y_test = generate_synthetic_data(n_samples=100, seed=123)

    # Train GDA model
    gda = GDA()
    gda.fit(X_train, y_train)

    # Display learned parameters
    print("\nLearned parameters:")
    print(f"φ (P(y=1)): {gda.phi:.4f}")
    print(f"μ_0: {gda.mu_0}")
    print(f"μ_1: {gda.mu_1}")
    print(f"Σ:\n{gda.sigma}")

    # Evaluate
    train_accuracy = gda.score(X_train, y_train)
    test_accuracy = gda.score(X_test, y_test)

    print(f"\nTraining accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")