import numpy as np
from decision_tree import ClassifierTree, RegressorTree


def bootstrap(X, y, rng):
    """Draw a bootstrap sample (sampling with replacement) of size n"""
    n = X.shape[0]
    indices = rng.integers(0, n, size=n)
    return X[indices], y[indices], indices

def oob_indices(n, bootstrap_indices):
    """Return indices that were NOT selected in a bootstrap draw"""
    selected = np.zeros(n, dtype=bool)
    selected[bootstrap_indices] = True
    return np.where(~selected)[0]

def random_best_split(tree, X, y, feature_indices):
    best_gain = -np.inf
    best_feature = None
    best_threshold = None

    for feature in feature_indices:
        for threshold in np.unique(X[:, feature]):
            left_mask, right_mask = tree.split(X, feature, threshold)

            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue

            gain = tree.impurity_reduction(y, left_mask, right_mask)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold


class BaseRandomForest:
    _tree_cls = None
    _default_max_features = "sqrt"

    def __init__(
            self,
            n_estimators=100,
            max_depth=None,
            min_split=2,
            max_features=None,
            random_state=None
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_split = min_split
        self.max_features = max_features
        self.random_state = random_state

        self.estimators_ = []
        self.feature_importance_ = []
        self.oob_score_ = None
        self._n_features = None
        self.n_classes = None

    def resolve_max_features(self, n_features: int) -> int:
        if self.max_features is not None:
            return int(self.max_features)
        if self._default_max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self._default_max_features == "third":
            return max(1, n_features // 3)
        return n_features

    def make_tree(self):
        return self._tree_cls(max_depth=self.max_depth, min_split=self.min_split)

    def grow_tree(self, X, y, k, rng):
        X_boot, y_boot, boot_idx = bootstrap(X, y, rng)
        tree = self.make_tree()

        def feature_selector(X_node, y_node):
            feature_indices = rng.choice(X.shape[1], size=k, replace=False)
            return random_best_split(tree, X_node, y_node, feature_indices)

        tree._feature_selector = feature_selector
        tree.fit(X_boot, y_boot)

        return tree, boot_idx

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n, n_features = X.shape

        self._n_features = n_features
        k = self.resolve_max_features(n_features)

        self._n_classes = len(np.unique(y))

        self.estimators_ = []
        boot_indices_list = []

        for _ in range(self.n_estimators):
            tree, boot_idx = self.grow_tree(X, y, k, rng)
            self.estimators_.append(tree)
            boot_indices_list.append(boot_idx)

        self.compute_feature_importance(n_features)
        self.compute_oob_score(X, y, boot_indices_list)

        return self

    def collect_pred(self, X):
        return np.array([tree.predict(X) for tree in self.estimators_])

    def predict(self, X):
        raise NotImplementedError

    def compute_feature_importance(self, n_features):
        importance = np.zeros(n_features)

        for tree in self.estimators_:
            stack = [(tree.root, 0)]
            while stack:
                node, depth = stack.pop()
                if node.value is not None:
                    continue

                importance[node.feature] += 1.0 / (depth + 1)
                stack.append((node.left, depth + 1))
                stack.append((node.right, depth + 1))

        total = sum(importance)
        self.feature_importance_ = importance / total if total > 0 else importance

    def compute_oob_score(self, X, y, boot_indices_list):
        n = X.shape[0]
        oob_preds = [[] for _ in range(n)]

        for tree, boot_idx in zip(self.estimators_, boot_indices_list):
            oob_idx = oob_indices(n, boot_idx)
            if len(oob_idx) == 0:
                continue
            for i, pred in zip(oob_idx, tree.predict(X[oob_idx])):
                oob_preds[i].append(pred)

        valid_idx = [i for i in range(n) if len(oob_preds[i]) > 0]
        if not valid_idx:
            self.oob_score_ = None
            return

        y_true = y[valid_idx]
        y_oob  = np.array([self._aggregate(oob_preds[i]) for i in valid_idx])
        self.oob_score_ = self._score(y_true, y_oob)

    def _aggregate(self, preds):
        raise NotImplementedError

    def _score(self, y_true, y_pred):
        raise NotImplementedError


class RandomForestClassifier(BaseRandomForest):
    _tree_cls = ClassifierTree
    _default_max_features = "sqrt"

    def predict(self, X):
        """"Majority vote: return the class with the most votes per sample"""
        all_preds = self.collect_pred(X)
        n_samples = X.shape[0]
        result = np.empty(n_samples, dtype=int)

        for i in range(n_samples):
            votes = all_preds[:, i].astype(int)
            result[i] = np.argmax(np.bincount(votes))

        return result

    def predict_proba(self, X):
        """
        Class probability estimates averaged across all trees
        Requires class labels to be 0-indexed integers
        """
        all_preds = self.collect_pred(X)
        n_samples = X.shape[0]
        proba = np.zeros((n_samples, self._n_classes))

        for i in range(n_samples):
            votes = all_preds[:, i].astype(int)
            counts = np.bincount(votes, minlength=self._n_classes)
            proba[i] = counts / np.sum(counts)

        return proba

    def _aggregate(self, preds: list):
        votes = np.array(preds, dtype=int)
        return int(np.argmax(np.bincount(votes)))

    def _score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(y_true.astype(int) == y_pred.astype(int)))


class RandomForestRegressor(BaseRandomForest):
    _tree_cls = RegressorTree
    _default_max_features = "third"

    def predict(self, X):
        """Average predictions across all trees"""
        return self.collect_pred(X).mean(axis=0)

    def _aggregate(self, preds: list) -> float:
        return float(np.mean(preds))

    def _score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0