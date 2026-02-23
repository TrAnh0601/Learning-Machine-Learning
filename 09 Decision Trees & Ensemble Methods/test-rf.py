import numpy as np
from random_forrest import RandomForestClassifier, RandomForestRegressor


def print_result(case_name, y_true, y_pred, oob=None, is_clf=True):
    print(f"\nCase: {case_name}")
    if is_clf:
        acc = np.mean(y_true == y_pred)
        print(f"  Accuracy : {acc:.2f} | Predictions: {y_pred}")
    else:
        mse = np.mean((y_true - y_pred) ** 2)
        print(f"  MSE      : {mse:.4f} | Avg Prediction: {np.mean(y_pred):.2f}")
    if oob is not None:
        print(f"  OOB Score: {oob:.4f}")


# Classifier

def test_classifier():
    print("CLASSIFIER TESTS")

    # 1. Perfect separation: RF phải đạt 100%
    X = np.array([[i, 0] for i in range(30)] + [[i, 10] for i in range(30)])
    y = np.array([0] * 30 + [1] * 30)
    clf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    print_result("Perfect Linear Separation", y, clf.predict(X), clf.oob_score_)

    # 2. Homogeneous labels: mọi tree phải trả về label duy nhất đó
    X2 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y2 = np.array([1, 1, 1, 1])
    clf2 = RandomForestClassifier(n_estimators=5, random_state=0).fit(X2, y2)
    print_result("Homogeneous Labels", y2, clf2.predict(X2), clf2.oob_score_)

    # 3. Conflicting features: điểm giống hệt nhau nhưng label khác
    # Expect: tree vẫn không crash, trả về majority class (0)
    X3 = np.array([[1, 1]] * 6)
    y3 = np.array([0, 0, 0, 1, 1, 0])
    clf3 = RandomForestClassifier(n_estimators=10, random_state=1).fit(X3, y3)
    print_result("Conflicting Features (majority=0)", y3, clf3.predict(X3), clf3.oob_score_)

    # 4. 3-class multiclass
    rng = np.random.default_rng(42)
    X4 = rng.standard_normal((90, 4))
    y4 = np.repeat([0, 1, 2], 30)
    X4 += y4.reshape(-1, 1) * 3  # separate classes
    clf4 = RandomForestClassifier(n_estimators=30, random_state=0).fit(X4, y4)
    print_result("3-class Multiclass", y4, clf4.predict(X4), clf4.oob_score_)

    # 5. max_depth=1 (stumps): kiểm tra RF không crash khi tree rất shallow
    clf5 = RandomForestClassifier(n_estimators=10, max_depth=1, random_state=0).fit(X4, y4)
    print_result("Shallow Trees (max_depth=1)", y4, clf5.predict(X4), clf5.oob_score_)

    # 6. Determinism
    c1 = RandomForestClassifier(n_estimators=10, random_state=7).fit(X4, y4)
    c2 = RandomForestClassifier(n_estimators=10, random_state=7).fit(X4, y4)
    match = np.array_equal(c1.predict(X4), c2.predict(X4))
    print(f"\nCase: Determinism (same seed)")
    print(f"  Predictions identical: {match}")

    # 7. Feature importance sums to 1
    print(f"\nCase: Feature Importance (sum == 1.0)")
    print(f"  Sum: {clf4.feature_importance_.sum():.6f} | Values: {np.round(clf4.feature_importance_, 3)}")

    # 8. predict_proba rows sum to 1
    proba = clf4.predict_proba(X4)
    print(f"\nCase: predict_proba row sums")
    print(f"  All rows sum to 1: {np.allclose(proba.sum(axis=1), 1.0)}")
    print(f"  Shape: {proba.shape}")


# Regressor

def test_regressor():
    print("\nREGRESSOR TESTS")

    # 1. Linear signal: expect low MSE
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 4))
    y = X[:, 0] * 3 + X[:, 1] * (-2) + rng.standard_normal(100) * 0.5
    reg = RandomForestRegressor(n_estimators=30, random_state=0).fit(X, y)
    print_result("Linear Signal", y, reg.predict(X), reg.oob_score_, is_clf=False)

    # 2. Constant target: MSE phải = 0
    X2 = rng.standard_normal((30, 3))
    y2 = np.full(30, 7.5)
    reg2 = RandomForestRegressor(n_estimators=5, random_state=0).fit(X2, y2)
    print_result("Constant Target (MSE should=0)", y2, reg2.predict(X2), reg2.oob_score_, is_clf=False)

    # 3. Outlier sensitivity
    X3 = np.linspace(0, 10, 20).reshape(-1, 1)
    y3 = X3.flatten() * 2
    y3[-1] = 500
    reg3 = RandomForestRegressor(n_estimators=20, random_state=0).fit(X3, y3)
    print_result("Outlier Sensitivity", y3, reg3.predict(X3), reg3.oob_score_, is_clf=False)

    # 4. Non-linear (cubic): RF nên fit tốt hơn linear model
    X4 = np.linspace(-2, 2, 40).reshape(-1, 1)
    y4 = X4.flatten() ** 3
    reg4 = RandomForestRegressor(n_estimators=30, max_depth=6, random_state=0).fit(X4, y4)
    print_result("Non-linear Cubic", y4, reg4.predict(X4), reg4.oob_score_, is_clf=False)

    # 5. More estimators → smoother predictions (lower variance)
    reg_small = RandomForestRegressor(n_estimators=3,  random_state=0).fit(X4, y4)
    reg_large = RandomForestRegressor(n_estimators=50, random_state=0).fit(X4, y4)
    print(f"\nCase: Variance Reduction (more trees)")
    print(f"  Pred variance  3 trees: {np.var(reg_small.predict(X4)):.4f}")
    print(f"  Pred variance 50 trees: {np.var(reg_large.predict(X4)):.4f}")


if __name__ == "__main__":
    try:
        test_classifier()
        test_regressor()
    except Exception as e:
        import traceback
        print(f"\nExecution Error: {e}")
        traceback.print_exc()