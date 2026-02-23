import numpy as np


class DecisionStump:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.polarity = 1

    def fit(self, X, y, sample_weight):
        n_samples, n_features = X.shape
        best_weighted_error = np.inf

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                for polarity in [1, -1]:
                    preds = np.where(polarity * X[:, feature] <= polarity * threshold, 1, -1)
                    weighted_error = np.sum(sample_weight[preds != y])

                    if weighted_error < best_weighted_error:
                        best_weighted_error = weighted_error
                        self.feature = feature
                        self.threshold = threshold
                        self.polarity = polarity
        return self

    def predict(self, X):
        return np.where(self.polarity * X[:, self.feature] <= self.polarity * self.threshold, 1, -1)


class AdaBoost:
    def __init__(self, n_estimators):
        self.n_estimators = n_estimators
        self.stumps = []
        self.alphas = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        D = np.full(n_samples, 1 / n_samples)

        for _ in range(self.n_estimators):
            # Resampling
            #indices = np.random.choice(n_samples, size=n_samples, replace=True, p=D)
            #X_resampled, y_resampled = X[indices], y[indices]

            # Train stump
            stump = DecisionStump()
            stump.fit(X, y, sample_weight=D) # sample_weight=np.ones(n_samples) / n_samples if using resampling

            # Evaluate on weighted distribution
            preds = stump.predict(X)
            eps = np.sum(D[preds != y])

            eps = np.clip(eps, 1e-10, 1 - 1e-10)
            alpha = 0.5 * np.log((1 - eps) / eps)

            # Update D
            D = D * np.exp(-alpha * y * preds)
            D /= np.sum(D)

            self.stumps.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        preds = sum(alpha * stump.predict(X) for alpha, stump in zip(self.alphas, self.stumps))
        return np.sign(preds)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


if __name__ == "__main__":
    # Sanity check
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import AdaBoostClassifier

    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    y = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = AdaBoost(n_estimators=50)
    model.fit(X_train, y_train)
    print(f"Custom AdaBoost accuracy: {model.score(X_test, y_test):.4f}")

    # Sklearn reference (weighted loss)
    ref = AdaBoostClassifier(n_estimators=50, random_state=42)
    ref.fit(X_train, np.where(y_train == -1, 0, 1))
    print(f"Sklearn AdaBoost accuracy:  {ref.score(X_test, np.where(y_test == -1, 0, 1)):.4f}")