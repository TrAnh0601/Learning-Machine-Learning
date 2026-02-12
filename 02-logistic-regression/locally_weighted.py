import numpy as np


def add_bias(X):
    ones = np.ones((X.shape[0], 1))
    return np.concatenate((ones, X), axis=1)


class LocallyWeighted:
    """
    Locally Weighted Regression (LWR) implementation.

    For each query point, fits a weighted linear regression where points
    closer to the query point receive higher weights.
    """
    def __init__(self, tau=0.5, ridge=1e-5):
        self.tau = tau
        self.ridge = ridge
        self.X_train = None
        self.y_train = None
        self.Xb = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.X_train = X
        self.y_train = y.ravel()
        self.Xb = add_bias(self.X_train)

        return self

    def _compute_weights(self, query_point):
        # Calculate squared Euclidean distance
        distance = np.sum((self.X_train - query_point) ** 2, axis=1)

        # Apply Gaussian kernel
        weights = np.exp(-distance / (2 * self.tau ** 2))

        return weights

    def _predict_single(self, query_point):
        weights = self._compute_weights(query_point)

        W_X = self.Xb * weights[:, None]
        Xt_wX = self.Xb.T @ W_X
        Xt_wy = self.Xb.T @ (weights * self.y_train)

        # Ridge regularization
        Xt_wX.flat[::len(Xt_wX) + 1] += self.ridge  # add to diagonal
        Xt_wX[0, 0] -= self.ridge  # remove from bias term

        try:
            theta = np.linalg.solve(Xt_wX, Xt_wy)
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(Xt_wX) @ Xt_wy

        return theta[0] + np.dot(theta[1:], query_point)

    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        predictions = np.array([self._predict_single(x) for x in X])

        return predictions


# Example usage and demonstration
if __name__ == "__main__":
    # Generate sample data: noisy sine wave
    np.random.seed(42)
    X_train = np.linspace(0, 10, 50)
    y_train = np.sin(X_train) + np.random.normal(0, 0.2, 50)

    # Create test points
    X_test = np.linspace(0, 10, 200)

    # Small tau (more local, wiggly fit)
    lwr_small = LocallyWeighted(tau=0.1)
    lwr_small.fit(X_train, y_train)
    y_pred_small = lwr_small.predict(X_test)

    # Medium tau (balanced)
    lwr_medium = LocallyWeighted(tau=0.5)
    lwr_medium.fit(X_train, y_train)
    y_pred_medium = lwr_medium.predict(X_test)

    # Large tau (more global, smooth fit)
    lwr_large = LocallyWeighted(tau=1.0)
    lwr_large.fit(X_train, y_train)
    y_pred_large = lwr_large.predict(X_test)

    print(f"\nPredictions generated for {len(X_test)} test points")
    print(f"Training data: {len(X_train)} points")
    print(f"\nSample predictions at X=5.0:")
    print(f"  tau=0.1: {lwr_small.predict([5.0])[0]:.4f}")
    print(f"  tau=0.5: {lwr_medium.predict([5.0])[0]:.4f}")
    print(f"  tau=1.0: {lwr_large.predict([5.0])[0]:.4f}")
    print(f"  True value: {np.sin(5.0):.4f}")