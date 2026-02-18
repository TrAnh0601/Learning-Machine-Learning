import numpy as np
from itertools import product


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


def cross_val_score(model, X, y, cv=5, scoring=None, fit_params=None, random_state=None):
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

        from copy import deepcopy
        fold_model = deepcopy(model)
        fold_model.fit(X_train, y_train, **fit_params)

        if scoring is not None:
            score = scoring(fold_model, X_val, y_val)
        else:
            score = fold_model.score(X_val, y_val)

        scores.append(score)

    return np.array(scores)


class GridSearchCV:
    def __init__(self, model, param_grid, cv=5, scoring=None, fit_params=None, random_state=None):
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.fit_params = fit_params or {}
        self.random_state = random_state

        # Set after fit()
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.best_model_ = None
        self.results_ = []

    def fit(self, X, y):
        self.results_ = []

        for params in self._get_param_combinations():
            from copy import deepcopy
            test_model = deepcopy(self.model)

            for param_name, param_value in params.items():
                setattr(test_model, param_name, param_value)

            cv_scores = cross_val_score(
                test_model, X, y,
                cv=self.cv,
                scoring=self.scoring,
                random_state=self.random_state
            )

            mean_score = cv_scores.mean()
            std_score = cv_scores.std()

            self.results_.append({
                'params': params,
                'mean_score': mean_score,
                'std_score': std_score,
                'fold_scores': cv_scores
            })

            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params

        from copy import deepcopy
        self.best_model_ = deepcopy(self.model)
        for param_name, param_value in self.best_params_.items():
            setattr(self.best_model_, param_name, param_value)
        self.best_model_.fit(X, y)

        return self

    def predict(self, X):
        if self.best_model_ is None:
            raise RuntimeError("Must call fit() before predict()")
        return self.best_model_.predict(X)

    def score(self, X, y):
        if self.best_model_ is None:
            raise RuntimeError("Must call fit() before score()")
        return self.best_model_.score(X, y)

    def print_results(self, top_n=10):
        if not self.results_:
            print("No results yet. Call fit() first.")
            return

        sorted_results = sorted(
            self.results_,
            key=lambda r: r['mean_score'],
            reverse=True
        )

        print(f"\n{'Rank':<6} {'Mean Score':>12} {'Std Dev':>10}   Parameters")
        print("-" * 70)

        for rank, result in enumerate(sorted_results[:top_n], 1):
            print(f"{rank:<6} {result['mean_score']:>12.4f} "
                  f"{result['std_score']:>10.4f}   {result['params']}")

        print(f"\nBest parameters: {self.best_params_}")
        print(f"Best CV score: {self.best_score_:.4f}")

    def _get_param_combinations(self):
        param_names = list(self.param_grid.keys())
        param_values = [self.param_grid[name] for name in param_names]

        for combination in product(*param_values):
            yield dict(zip(param_names, combination))