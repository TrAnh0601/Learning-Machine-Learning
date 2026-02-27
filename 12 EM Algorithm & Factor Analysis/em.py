import numpy as np


class GaussianMixture:
    def __init__(self, K, max_iter=100, tol=1e-6, random_state=None):
        self.K = K
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.default_rng(random_state)

    def _init_params(self, X):
        m, n = X.shape

        # phi: uniform prior over clusters
        self.phi = np.full(self.K, 1.0 / self.K)

        # mu: random data points as initial centroids
        idx = self.rng.choice(m, self.K, replace=False)
        self.mu = X[idx].copy()

        # sigma: K copies of empirical covariance - avoids degenerate start
        emp_cov = np.cov(X, rowvar=False)
        self.sigma = np.stack([emp_cov.copy() for _ in range(self.K)])

    def _gaussian_log_pdf(self, X, mu, sigma):
        n = X.shape[1]
        diff = X - mu
        sign, log_det = np.linalg.slogdet(sigma)

        if sign <= 0:
            raise ValueError("Sigma must be positive")

        sigma_inv = np.linalg.inv(sigma)
        mahal = np.einsum("mi,ij,mj->m", diff, sigma_inv, diff)
        return -0.5 * (n * np.log(2 * np.pi) + log_det + mahal)

    def _e_step(self, X):
        m = X.shape[0]
        log_w = np.zeros((m, self.K))

        for j in range(self.K):
            log_w[:, j] = np.log(self.phi[j] + 1e-10) + self._gaussian_log_pdf(X, self.mu[j], self.sigma[j])

        # normalization
        log_w -= log_w.max(axis=1, keepdims=True)
        w = np.exp(log_w)
        w /= w.sum(axis=1, keepdims=True)
        return w

    def _m_step(self, X, w):
        m, n = X.shape
        w_sum = w.sum(axis=0)

        self.phi = w_sum / m

        for j in range(self.K):
            w_j = w[:, j]

            # mu update
            self.mu[j] = (w_j @ X) / w_sum[j]

            # sigma update
            diff = X - self.mu[j]
            self.sigma[j] = (w_j * diff.T) @ diff / w_sum[j]

            # regularization
            self.sigma[j] += 1e-6 * np.eye(n)

    def _log_likelihood(self, X):
        m = X.shape[0]
        log_probs = np.zeros((m, self.K))
        for j in range(self.K):
            log_probs[:, j] = np.log(self.phi[j] + 1e-300) + self._gaussian_log_pdf(X, self.mu[j], self.sigma[j])

        # log-sum-exp over K
        max_log = log_probs.max(axis=1, keepdims=True)
        log_sum = np.log(np.exp(log_probs - max_log).sum(axis=1)) + max_log.squeeze()
        return log_sum.sum()

    def fit(self, X, verbose=False):
        self._init_params(X)
        self.log_likelihoods_ = []
        prev_ll = -np.inf

        for t in range(self.max_iter):
            w = self._e_step(X)
            self._m_step(X, w)

            ll = self._log_likelihood(X)
            self.log_likelihoods_.append(ll)

            if verbose:
                print(f"Iter: {t+1:3d}, ll: {ll:.4f}")

            # Convergence check
            if abs(ll - prev_ll) < self.tol:
                if verbose:
                    print(f"Converged at iteration {t+1:3d}")
                break
            prev_ll = ll

        self.w_ = w
        return self

    def predict(self, X):
        w = self._e_step(X)
        return w.argmax(axis=1)

    def predict_proba(self, X):
        return self._e_step(X)


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Ground-truth GMM: 3 clusters in 2D
    K_true = 3
    mu_true = np.array([[0, 0], [5, 5], [-5, 5]], dtype=float)
    sizes = [100, 120, 80]

    X = np.vstack([
        rng.multivariate_normal(mu_true[j], np.eye(2), sizes[j])
        for j in range(K_true)
    ])
    labels_true = np.hstack([[j] * s for j, s in enumerate(sizes)])

    # Fit
    model = GaussianMixture(K=3, max_iter=100, tol=1e-6, random_state=0)
    model.fit(X, verbose=True)

    # Verify monotone non-decreasing (EM guarantee)
    lls = model.log_likelihoods_
    assert all(lls[i+1] >= lls[i] - 1e-6 for i in range(len(lls)-1)), \
        "BUG: log-likelihood decreased — violates EM guarantee"

    print(f"\nLearned mu:\n{model.mu}")
    print(f"Learned phi: {model.phi}")
    print(f"Final log-likelihood: {lls[-1]:.4f}")

    # Cluster accuracy (permutation-invariant, rough check)
    preds = model.predict(X)
    print(f"Unique predicted clusters: {np.unique(preds)}")