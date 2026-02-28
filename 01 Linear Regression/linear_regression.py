import numpy as np

# UTILITY FUNCTIONS
def add_bias(X):
    """Add intercept term (column of ones) to feature matrix"""
    ones = np.ones((X.shape[0], 1))
    return np.concatenate((ones, X), axis=1)


class LinearRegression:
    def __init__(self, regularization=None, lambda_1=0.0, lambda_2=0.0):
        self.w = None
        self.regularization = regularization
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def fit_normal(self, X, y):
        """Fit using Normal Equations (closed-form solution)"""
        Xb = add_bias(X)
        y = y.reshape(-1, 1)

        if self.regularization == 'l2':
            # Ridge: (X^T X + λI)^(-1) X^T y
            n = Xb.shape[1]
            reg_matrix = self.lambda_1 * np.eye(n)
            reg_matrix[0, 0] = 0
            self.w = np.linalg.inv(Xb.T @ Xb + reg_matrix) @ Xb.T @ y

        elif self.regularization == 'l1':
            raise ValueError("Lasso (L1) has no closed-form solution")

        elif self.regularization == 'elasticnet':
            raise ValueError("Elastic Net has no closed-form solution")

        else:
            # Standard linear regression
            self.w = np.linalg.pinv(Xb) @ y

        return self

    def fit_gd(self, X, y, learning_rate=0.01, n_iterations=1000,
               early_stopping=False, patience=100):
        Xb = add_bias(X)
        m, n = Xb.shape
        y = y.reshape(-1, 1)

        self.w = np.zeros((n, 1), dtype=float)
        best_loss = np.inf
        wait = 0

        for _ in range(n_iterations):
            y_hat = Xb @ self.w
            error = y_hat - y

            gradient = (Xb.T @ error) / m

            if self.regularization == 'l2':
                reg_grad = (self.lambda_1 / m) * self.w
                reg_grad[0, 0] = 0
                gradient += reg_grad
            elif self.regularization == 'l1':
                reg_grad = (self.lambda_1 / m) * np.sign(self.w)
                reg_grad[0, 0] = 0
                gradient += reg_grad
            elif self.regularization == 'elasticnet':
                reg_grad = (self.lambda_1 / m) * self.w + (self.lambda_2 / m) * np.sign(self.w)
                reg_grad[0, 0] = 0
                gradient += reg_grad

            self.w -= learning_rate * gradient
            loss = self._compute_cost(Xb, y)

            if early_stopping:
                if loss < best_loss:
                    best_loss = loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        print('Early stopping triggered.')
                        break

        return self

    def _compute_cost(self, Xb, y):
        """Compute cost function J(w) including regularization."""
        m = Xb.shape[0]
        y_hat = Xb @ self.w

        # MSE cost
        cost = (1 / (2 * m)) * np.sum((y_hat - y) ** 2)

        if self.regularization == 'l2':
            cost += (self.lambda_1 / (2 * m)) * np.sum(self.w[1:] ** 2)
        elif self.regularization == 'l1':
            cost += (self.lambda_1 / m) * np.sum(np.abs(self.w[1:]))
        elif self.regularization == 'elasticnet':
            cost += (
                (self.lambda_1 / (2 * m)) * np.sum(self.w[1:] ** 2) +
                (self.lambda_2 / m) * np.sum(np.abs(self.w[1:]))
            )

        return cost

    def predict(self, X):
        if self.w is None:
            raise ValueError("Model not fitted yet")
        Xb = add_bias(X)
        return (Xb @ self.w).reshape(-1)


class Ridge(LinearRegression):
    def __init__(self, alpha=1.0):
        super().__init__(regularization='l2', lambda_1=alpha)


class Lasso(LinearRegression):
    def __init__(self, alpha=1.0):
        super().__init__(regularization='l1', lambda_1=alpha)


class ElasticNet(LinearRegression):
    def __init__(self, alpha_1=1.0, alpha_2=1.0):
        super().__init__(regularization='elasticnet', lambda_1=alpha_1, lambda_2=alpha_2)


if __name__ == "__main__":
    # Test on synthetic data
    np.random.seed(42)
    X = np.random.randn(10, 1)
    y = 3 + 2 * X.ravel() + 0.5 * np.random.randn(10)

    # Standard Linear Regression
    lr = LinearRegression()
    lr.fit_normal(X, y)
    print(f"LR Weight: {lr.w[1:]}, {lr.w[0, 0]}")

    # Ridge Regression
    ridge = Ridge(alpha=0.1)
    ridge.fit_gd(X, y, learning_rate=0.1, n_iterations=1000)
    print(f"Ridge Weight: {ridge.w[1:]}, {ridge.w[0, 0]}")

    # Lasso Regression
    lasso = Lasso(alpha=0.1)
    lasso.fit_gd(X, y, learning_rate=0.1, n_iterations=1000)
    print(f"Lasso Weight: {lasso.w[1:]}, {lasso.w[0, 0]}")

    # Elastic Net Regression
    elastic_net = ElasticNet(alpha_1=0.5, alpha_2=0.1)
    elastic_net.fit_gd(X, y, learning_rate=0.1, n_iterations=1000)
    print(f"Elastic Net Weight: {lasso.w[1:]}, {lasso.w[0, 0]}")