import numpy as np

# MODEL
def add_bias(X):
    ones = np.ones((X.shape[0], 1))
    return np.concatenate((ones, X), axis=1)

def mse(y, y_hat):
    y = np.array(y)
    y_hat = np.array(y_hat)
    y = y.reshape(-1)
    y_hat = y_hat.reshape(-1)
    return np.mean((y - y_hat) ** 2)


class LinearRegression:
    def __init__(self):
        self.w = None

    def fit_normal(self, X, y):
        Xb = add_bias(X)
        y = y.reshape(-1, 1)
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

            self.w -= learning_rate * gradient
            loss = mse(y, y_hat)

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

    def predict(self, X):
        Xb = add_bias(X)
        return Xb @ self.w


# TEST
np.random.seed(0)

# Generate data
X = np.linspace(0, 10, 50).reshape(-1, 1)
y = 2 + 3 * X + np.random.randn(50, 1) * 0.5  # small noise

# Test normal
model_normal = LinearRegression()
model_normal.fit_normal(X, y)

print("Normal equation weights:")
print(model_normal.w)

# Test gradient descent
model_gd = LinearRegression()
model_gd.fit_gd(
    X, y,
    learning_rate=0.01,
    n_iterations=5000,
    early_stopping=True,
    patience=200
)

print("Gradient descent weights:")
print(model_gd.w)

# Test prediction
X_test = np.array([[0], [5], [10]])
y_pred = model_gd.predict(X_test)

print("Predictions:")
for x, yp in zip(X_test.flatten(), y_pred.flatten()):
    print(f"x={x:.1f}, ŷ={yp:.2f}")