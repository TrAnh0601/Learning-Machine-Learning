# Chapter: PCA & ICA


## 1. PCA (Principal Component Analysis)

### Goal
Find a $k$-dimensional subspace ($k \ll n$) that **preserves maximum variance** from the original data.

### Setup
- Dataset $\{x^{(i)}\}_{i=1}^m$, $x^{(i)} \in \mathbb{R}^n$, mean-centered.
- Empirical covariance matrix: $\Sigma = \frac{1}{m} \sum_i x^{(i)} x^{(i)\top}$

### Two Equivalent Views

| View | Formulation |
|---|---|
| Maximize variance | $\max_{\|u\|=1} \ u^\top \Sigma u$ |
| Minimize reconstruction error | $\min_{U} \ \frac{1}{m}\sum_i \|x^{(i)} - UU^\top x^{(i)}\|^2$ |

Both lead to the same solution: **eigenvectors of $\Sigma$**.

### Solution
$$\Sigma u = \lambda u$$

Select the $k$ eigenvectors corresponding to the $k$ **largest** eigenvalues → principal components $U_k \in \mathbb{R}^{k \times n}$.

### Algorithm
1. Mean-center: $x \leftarrow x - \mu$
2. Compute $\Sigma = \frac{1}{m} X^\top X$
3. Eigendecomposition of $\Sigma$ (or SVD of $X$ directly — more numerically stable)
4. Project: $z = U_k^\top x \in \mathbb{R}^k$
5. Reconstruct: $\hat{x} = U_k z + \mu$

### Choosing $k$
Pick the smallest $k$ such that:
$$\frac{\sum_{j=1}^k \lambda_j}{\sum_{j=1}^n \lambda_j} \geq 0.99$$

### Limitations
- Captures **linear** structure only.
- Uses only **2nd-order statistics** — blind to higher-order dependencies.
- Components have no semantic meaning; they are directions of variance.


## 2. ICA (Independent Component Analysis)

### Goal
**Recover independent source signals** from observed mixtures.

### Problem Setup (Cocktail Party)
$$x = As$$

- $s \in \mathbb{R}^n$: unknown independent sources
- $A \in \mathbb{R}^{n \times n}$: unknown mixing matrix
- $x \in \mathbb{R}^n$: observed signals

**Goal:** recover $W = A^{-1}$ (unmixing matrix) from observations $\{x^{(i)}\}$ alone.

### Key Identifiability Condition
$A$ and $s$ are recoverable (up to permutation and scaling) **if and only if at most one source is Gaussian.**

> **Why Gaussian fails:** If $s \sim \mathcal{N}(0, I)$, any rotation $R$ gives $AR$ the same likelihood — non-identifiable.

ICA exploits **non-Gaussianity** of sources to break this ambiguity.

### CS229 Formulation — MLE
Assume sources $s_j \sim p_s(s_j)$ i.i.d., non-Gaussian. The log-likelihood is:

$$\ell(W) = \sum_{i=1}^m \left[ \sum_{j=1}^n \log p_s(w_j^\top x^{(i)}) + \log |\det W| \right]$$

**Stochastic gradient ascent update:**
$$W \leftarrow W + \alpha \left( \begin{bmatrix} 1 - 2g(w_1^\top x) \\ \vdots \\ 1 - 2g(w_n^\top x) \end{bmatrix} x^\top + (W^\top)^{-1} \right)$$

where $g(s) = 1 - 2\sigma(s)$ is the score function from the **logistic prior** (assumes super-Gaussian sources).

### Source Distribution Prior
CS229 uses a logistic-based prior whose CDF is sigmoid:
$$p_s(s) \propto \sigma(s)(1 - \sigma(s))$$

Assumes **super-Gaussian** sources (heavy-tailed, excess kurtosis $> 0$), e.g., speech, natural images.  
If sources are sub-Gaussian (e.g., uniform), this prior will fail.

### Algorithm
1. Mean-center $X$
2. **Whiten:** $\tilde{X} = \Sigma^{-1/2} X$ so that $\text{Cov}(\tilde{X}) = I$
3. Initialize $W$ randomly (orthogonalize via QR)
4. Run stochastic gradient ascent on $\ell(W)$
5. Re-orthogonalize $W$ each epoch: $W \leftarrow (WW^\top)^{-1/2} W$
6. Recover sources: $s = Wx$

## 3. PCA vs ICA — Summary

| | PCA                      | ICA                     |
|---|--------------------------|-------------------------|
| Objective | Maximize variance        | Maximize independence   |
| Statistics used | 2nd order (covariance)   | Higher-order            |
| Works with Gaussian sources | Yes                      | No                      |
| Output components | Uncorrelated             | Independent             |
| Unique solution | Up to sign/order         | Up to permutation/scale |
| Typical use | Dimensionality reduction | Source separation       |

### Key Relationship
**PCA (whitening) is standard preprocessing for ICA.**  
Whitening reduces the search space to orthogonal matrices, improving convergence and numerical stability.