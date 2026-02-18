import numpy as np
from cross_validation import cross_val_score, GridSearchCV, KFold, StratifiedKFold
from logistic_regression import LogisticRegression

np.random.seed(42)
X = np.random.randn(500, 5)
y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)

X_train, X_test = X[:400], X[400:]
y_train, y_test = y[:400], y[400:]

print("BASIC TRAINING")
model = LogisticRegression()
model.fit(X_train, y_train, learning_rate=0.1, n_iters=1000)
print(f"Train Accuracy: {model.score(X_train, y_train):.3f}")
print(f"Test Accuracy: {model.score(X_test, y_test):.3f}\n")

print("K-FOLD CROSS-VALIDATION")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=kfold)
print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.3f}, Std: {scores.std():.3f}\n")

print("STRATIFIED K-FOLD")
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=skfold)
print(f"Stratified CV Scores: {scores}")
print(f"Mean: {scores.mean():.3f}, Std: {scores.std():.3f}\n")

print("GRID SEARCH")
param_grid = {
    'C': [0.1, 1.0, 10.0],
    'tol': [1e-4, 1e-3]
}

grid = GridSearchCV(
    LogisticRegression(),
    param_grid,
    cv=5,
    random_state=42
)

grid.fit(X_train, y_train)
grid.print_results(top_n=5)

print(f"Final Test Accuracy: {grid.score(X_test, y_test):.3f}\n")

print("GRID SEARCH WITH FIT PARAMETERS")

param_grid = {
    'C': [0.1, 1.0, 10.0]
}

grid_with_fit = GridSearchCV(
    LogisticRegression(),
    param_grid,
    cv=3,
    fit_params={'learning_rate': 0.2, 'n_iters': 500},
    random_state=42
)

grid_with_fit.fit(X_train, y_train)
print(f"Best C: {grid_with_fit.best_params_}")
print(f"Best Score: {grid_with_fit.best_score_:.3f}")