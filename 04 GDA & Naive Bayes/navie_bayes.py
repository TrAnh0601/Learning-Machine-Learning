import numpy as np


class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_priors = None
        self.feature_probs = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes) # For the spam classification problem in the lecture, n_classes == 2

        self.class_priors = np.zeros(n_classes)
        self.feature_probs = np.zeros((n_samples, n_features, 2))

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_priors[idx] = X_c.shape[0] / n_samples

            for j in range(n_features):
                self.feature_probs[idx, j, 1] = (np.sum(X_c[:, j] == 1) + 1) / (X_c.shape[0] + 2)
                self.feature_probs[idx, j, 0] = (np.sum(X_c[:, j] == 0) + 1) / (X_c.shape[0] + 2)

    def predict(self, X):
        predictions = []

        for x in X:
            posteriors = []

            for idx, c in enumerate(self.classes):
                prior = np.log(self.class_priors[idx])

                likelihood = 0
                for j, feature_val in enumerate(x):
                    likelihood += np.log(self.feature_probs[idx, j, int(feature_val)])

                posterior = prior + likelihood
                posteriors.append(posterior)

            predictions.append(self.classes[np.argmax(posteriors)])

        return np.array(predictions)

# TEST
if __name__ == "__main__":
    # Simple binary classification example
    X_train = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
        [0, 1, 0]
    ])

    y_train = np.array([1, 1, 0, 0, 1, 0])

    nb = NaiveBayes()
    nb.fit(X_train, y_train)

    # Test data
    X_test = np.array([
        [1, 1, 0],
        [0, 0, 1]
    ])

    predictions = nb.predict(X_test)
    print("Predictions:", predictions)

    train_predictions = nb.predict(X_train)
    accuracy = np.mean(train_predictions == y_train)
    print(f"Training accuracy: {accuracy:.2f}")
