import numpy as np


class KFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        m = X.shape[0]
        indices = np.arange(m)

        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)

        folds = np.array_split(indices, self.n_splits)

        for i in range(self.n_splits):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(self.n_splits) if j != i])
            yield train_idx, val_idx

    def get_n_splits(self):
        return self.n_splits


class StratifiedKFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        y = np.array(y).ravel()
        rng = np.random.default_rng(self.random_state)
        classes = np.unique(y)

        class_indices = []
        for cls in classes:
            idx = np.where(y == cls)[0]
            if self.shuffle:
                rng.shuffle(idx)
            class_indices.append(idx)

        class_folds = [np.array_split(idx, self.n_splits) for idx in class_indices]

        for i in range(self.n_splits):
            val_idx = np.concatenate([cf[i] for cf in class_folds])
            train_idx = np.concatenate([
                np.concatenate([cf[j] for j in range(self.n_splits) if j != i])
                for cf in class_folds
            ])
            yield train_idx, val_idx

    def get_n_splits(self):
        return self.n_splits


def cross_val_score(estimator, X, y, cv=5, scoring=None, fit_params=None, random_state=None):
    if fit_params is None:
        fit_params = {}

    if isinstance(cv, int):
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        splitter = cv

    scores = []
    for train_idx, val_idx in splitter.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        fold_est = estimator.__class__(**estimator.get_params())
        fold_est.fit(X_train, y_train, **fit_params)

        if scoring is not None:
            s = scoring(fold_est, X_val, y_val)
        else:
            s = fold_est.score(X_val, y_val)

        scores.append(s)

    return np.array(scores)