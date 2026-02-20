import numpy as np
from decision_tree import ClassifierTree, RegressorTree


def print_result(case_name, y_true, y_pred, is_clf=True):
    print(f"Case: {case_name}")
    if is_clf:
        acc = np.mean(y_true == y_pred)
        print(f"Accuracy: {acc:.2f} | Predictions: {y_pred}")
    else:
        mse = np.mean((y_true - y_pred) ** 2)
        print(f"MSE: {mse:.4f} | Avg Prediction: {np.mean(y_pred):.2f}")


def edge_cases_classifier():
    """Exploring ClassifierTree limits with boundary conditions."""

    # 1. Homogeneous Data: All samples have the same label
    X1 = np.array([[1, 2], [3, 4], [5, 6]])
    y1 = np.array([1, 1, 1])
    clf = ClassifierTree(max_depth=2).fit(X1, y1)
    print_result("Homogeneous Labels", y1, clf.predict(X1))

    # 2. Conflicting Features: Identical features but different labels
    # Tests if the tree handles non-separable points without infinite recursion
    X2 = np.array([[1, 1], [1, 1], [1, 1]])
    y2 = np.array([0, 1, 0])
    clf = ClassifierTree(max_depth=2).fit(X2, y2)
    print_result("Conflicting Features", y2, clf.predict(X2))

    # 3. XOR Pattern: Classic non-linear relationship
    # Tests the tree's ability to create orthogonal decision boundaries
    X3 = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    y3 = np.array([0, 0, 1, 1])
    clf = ClassifierTree(max_depth=5).fit(X3, y3)
    print_result("XOR Pattern", y3, clf.predict(X3))

    # 4. Redundant Features: Includes constant/useless columns
    # Checks if the best_split logic correctly ignores zero-variance features
    X4 = np.array([[1, 0, 99], [2, 0, 99], [1, 1, 99], [2, 1, 99]])
    y4 = np.array([0, 0, 1, 1])
    clf = ClassifierTree(max_depth=3).fit(X4, y4)
    print_result("Redundant/Constant Features", y4, clf.predict(X4))


def edge_cases_regressor():
    """Exploring RegressorTree limits with variance and outliers."""

    # 1. Extreme Outlier: Tests the sensitivity of Sum of Squared Residuals (SSR)
    X1 = np.linspace(0, 10, 10).reshape(-1, 1)
    y1 = X1.flatten() * 2
    y1[-1] = 500  # Severe outlier
    reg = RegressorTree(max_depth=5).fit(X1, y1)
    print_result("Outlier Sensitivity", y1, reg.predict(X1), is_clf=False)

    # 2. Constant Target: Zero variance in the output
    X2 = np.random.rand(5, 2)
    y2 = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    reg = RegressorTree(max_depth=2).fit(X2, y2)
    print_result("Constant Target", y2, reg.predict(X2), is_clf=False)

    # 3. High-Degree Polynomial: Non-linear regression (Cubic function)
    X3 = np.linspace(-2, 2, 20).reshape(-1, 1)
    y3 = X3.flatten() ** 3
    reg = RegressorTree(max_depth=10).fit(X3, y3)
    print_result("Non-linear Cubic Function", y3, reg.predict(X3), is_clf=False)


if __name__ == "__main__":
    try:
        edge_cases_classifier()
        edge_cases_regressor()
    except Exception as e:
        print(f"Execution Error: {e}")
        print("Tip: Check if ClassifierTree.predict() in 'decision_tree.py' has square brackets [].")