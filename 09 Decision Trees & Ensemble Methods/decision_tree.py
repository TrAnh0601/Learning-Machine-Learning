import numpy as np
from abc import ABC, abstractmethod


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class BaseDecisionTree(ABC):
    def __init__(self, max_depth=None, min_split=2):
        self.max_depth = max_depth
        self.min_split = min_split
        self.root = None

    @abstractmethod
    def impurity(self, y):
        pass

    @abstractmethod
    def leaf_value(self, y):
        pass

    @abstractmethod
    def stop(self, y):
        pass

    def split(self, X, feature, threshold):
        left_mask = X[:, feature] <= threshold
        return left_mask, ~left_mask

    def impurity_reduction(self, y, left_mask, right_mask):
        n = len(y)
        n_left, n_right = left_mask.sum(), right_mask.sum()

        return (
            self.impurity(y)
            - (n_left / n) * self.impurity(y[left_mask])
            - (n_right / n) * self.impurity(y[right_mask])
        )

    def best_split(self, X, y):
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_mask, right_mask = self.split(X, feature, threshold)

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                gain = self.impurity_reduction(y, left_mask, right_mask)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0):
        n_samples = X.shape[0]

        stop = (
            (self.max_depth is not None and depth >= self.max_depth)
            or self.stop(y)
            or n_samples < self.min_split
        )
        if stop:
            return Node(value=self.leaf_value(y))

        # Use feature_selector in Random Forrest
        selector = getattr(self, "_feature_selector", None)
        best_feature, best_threshold = (
            selector(X, y) if selector is not None else self.best_split(X, y)
        )

        if best_feature is None:
            return Node(value=self.leaf_value(y))

        left_mask, right_mask = self.split(X, best_feature, best_threshold)

        left_child = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(best_feature, best_threshold, left_child, right_child)

    def fit(self, X, y):
        self.root = self.build_tree(X, y)
        return self

    def predict_single(self, x):
        node = self.root
        while node.value is None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])


class ClassifierTree(BaseDecisionTree):
    def impurity(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def leaf_value(self, y):
        return np.argmax(np.bincount(y.astype(int)))

    def stop(self, y):
        return len(np.unique(y)) == 1


class RegressorTree(BaseDecisionTree):
    def impurity(self, y):
        return float(np.var(y))

    def leaf_value(self, y):
        return np.mean(y).astype(float)

    def stop(self, y):
        return np.var(y).astype(float) <= 1e-10