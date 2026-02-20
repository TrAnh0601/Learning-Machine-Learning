import numpy as np


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=None, min_split=2):
        self.max_depth = max_depth
        self.min_split = min_split
        self.root = None

    def gini(self, y):
        _, n_classes = np.unique(y, return_counts=True)
        probs = n_classes / len(y)
        return 1 - np.sum(probs ** 2)

    def split(self, X, y, feature, threshold):
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        return (X[left_mask], y[left_mask]), (X[right_mask], y[right_mask])

    def information_gain(self, y, y_left, y_right):
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)

        gini_parent = self.gini(y)
        gini_children = (n_left / n) * self.gini(y_left) + (n_right / n) * self.gini(y_right)

        return gini_parent - gini_children

    def best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None

        N_features = X.shape[1]

        for feature in range(N_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                (_, y_left), (_, y_right) = self.split(X, y, feature, threshold)

                if len(y_left) == 0 or (y_right) == 0:
                    continue

                gain = self.information_gain(y, y_left, y_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        if (depth >= self.max_depth if self.max_depth else None) or \
            n_classes == 1 or \
            n_samples < self.min_split:
            leaf_value = np.argmax(np.bincount(y.astype(int)))
            return Node(value=leaf_value)

        best_feature, best_threshold = self.best_split(X, y)

        if best_feature is None:
            leaf_value = np.argmax(np.bincount(y.astype(int)))
            return Node(value=leaf_value)

        left_child = self.build_tree(X[best_feature], y[best_feature], depth + 1)
        right_child = self.build_tree(X[best_feature], y[best_feature], depth + 1)

        return Node(best_feature, best_threshold, left_child, right_child)

    def fit(self, X, y):
        self.root = self.build_tree(X, y)
        return self

    def predict_sample(self, X, node):
        if node.value is not None:
            return node.value

        if X[node.feature] <= node.threshold:
            return self.predict_sample(X, node.left)

        return self.predict_sample(X, node.right)

    def predict(self, X):
        return np.array(self.predict_sample(x, self.root) for x in X)