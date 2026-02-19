# Chapter: Learning Theory — Approx/Estimation Error & ERM

## 1. Introduction & Core Assumptions

Learning theory provides the mathematical framework for understanding **why machine learning algorithms generalize** from finite training data to unseen test data. Two foundational assumptions underpin most of the results:

- **Data Distribution (D):** A fixed, unknown distribution _D_ generates all _(x, y)_ pairs.
- **IID Sampling:** All training and test examples are drawn IID from _D_.

> **Insight: The Learning Process as a Random Variable** Since training set _S_ is a collection of _m_ random samples, it is a random variable. A deterministic learning algorithm fed _S_ produces hypothesis _ĥ_ (or _θ̂_) that is itself a random variable with its own sampling distribution. This is the foundation for all bias-variance analysis.

**IID in Practice — When It Breaks and Why It Matters:** IID is routinely violated in production systems. Recognizing _how_ it breaks determines which remediation applies:

|Violation|Example|Consequence|
|---|---|---|
|**Covariate shift**|Train on daytime images, deploy at night|_P(x)_ changes; _P(y\|x)_ stable. Importance weighting can correct.|
|**Label shift**|Class priors differ between train and test|_P(y)_ changes. Prior correction or target shift methods needed.|
|**Concept drift**|User behavior evolves over time|_P(y\|x)_ changes. Requires online learning or periodic retraining.|
|**Temporal correlation**|Time-series data, correlated residuals|IID assumption on samples fails — standard bounds no longer hold.|
|**Federated / non-IID splits**|Per-device data heterogeneity|Local ERM solutions diverge from global optimum; FedAvg may not converge.|


## 2. Bias-Variance Tradeoff: The Parameter View

Bias and variance are properties of the **sampling distribution of the estimator** _θ̂_, not just properties of the fitted curve.

$$\text{Bias}(\hat{\theta}) = E[\hat{\theta}] - \theta^* \qquad \text{Var}(\hat{\theta}) = E\left[(\hat{\theta} - E[\hat{\theta}])^2\right]$$

**Asymptotic Behavior** as _m_ → ∞:

- Variance → 0 (concentrates around its expectation).
- **Consistency:** _θ̂ →ᵖ θ*_ — estimator converges in probability to the truth.
- **Statistical Efficiency:** Rate of variance reduction relative to _m_. An efficient estimator achieves the Cramér-Rao lower bound; variance decreases as _O(1/m)_.

**The Classical vs. Modern Regime — Double Descent:**

Classical bias-variance intuition predicts a U-shaped test error curve as model complexity increases. Modern overparameterized models (neural networks, kernel methods at large scale) exhibit **double descent**: after the interpolation threshold (zero training error), test error _decreases again_ as parameters grow further. Mechanisms include:

- Implicit regularization from gradient descent (SGD finds minimum-norm solutions).
- Benign overfitting: interpolating noise without distorting signal when the signal lies in a low-dimensional subspace of a high-dimensional parameter space.
- The classical bias-variance decomposition remains valid, but variance can _decrease_ in the overparameterized regime due to averaging over many near-zero components.

**Implication for RE:** The classical tradeoff governs model selection in data-scarce regimes. In data-rich, large-model regimes, scaling laws (Hoffmann et al., Chinchilla) empirically characterize the optimal compute-data tradeoff — theory is catching up but practice has outpaced it.

## 3. Error Decomposition

The **Generalization Error (Risk)** is:

$$\varepsilon(h) = E_{(x,y) \sim D}[\mathbf{1}{h(x) \neq y}]$$

Decomposed into three components:

|Component|Formal Definition|Diagnostic Signal|
|---|---|---|
|**Bayes Error (Irreducible)**|_ε(G)_ where _G_ is the Bayes-optimal classifier|High train _and_ test error; irreducible via modeling|
|**Approximation Error**|_ε(h*) − ε(G)_, where _h*_ = best in _H_|Train error structurally bounded away from Bayes|
|**Estimation Error**|_ε(ĥ) − ε(h*)_|Gap between val and train error; shrinks with more data|

$$\varepsilon(\hat{h}) = \underbrace{\varepsilon(G)}_{\text{Bayes}} + \underbrace{[\varepsilon(h^*) - \varepsilon(G)]}_{\text{Approximation}} + \underbrace{[\varepsilon(\hat{h}) - \varepsilon(h^*)]}_{\text{Estimation}}$$

**Using This Decomposition as a Diagnostic Framework:**

When a model underperforms in production or experiments, this decomposition gives a structured path to root cause:

- **Bayes error dominant:** The task itself is noisy or ambiguous. Audit label quality, consider human-level performance as a ceiling, or reframe the task.
- **Approximation error dominant:** The model class is too restricted. Increase capacity (more layers, wider networks, richer features), or relax inductive biases. Common in early stages of a project using baseline architectures.
- **Estimation error dominant:** Classic overfitting. Options: increase _m_, apply regularization (L1/L2, dropout, weight decay), use data augmentation, or reduce model capacity.

**In research settings**, isolating approximation error is especially important. A common failure mode is over-attributing poor results to the optimizer or data pipeline when the hypothesis class itself cannot represent the target function.

## 4. Empirical Risk Minimization (ERM)

ERM selects _h ∈ H_ minimizing training error:

$$\hat{h} = \arg\min_{h \in H} \hat{\varepsilon}(h) = \arg\min_{h \in H} \frac{1}{m} \sum_{i=1}^{m} \mathbf{1}{h(x^{(i)}) \neq y^{(i)}}$$

**The central question:** Does minimizing _ε̂(h)_ guarantee low _ε(h)_?

The answer is conditional on **uniform convergence** — the property that _ε̂(h) ≈ ε(h)_ simultaneously for all _h ∈ H_. If this holds, the ERM solution _ĥ_ tracks the best-in-class _h*_ closely.

**ERM Failure Modes:**

|Failure Mode|Mechanism|Mitigation|
|---|---|---|
|**Distribution shift**|_P_train ≠ P_test_|Domain adaptation, importance weighting, robust optimization (DRO)|
|**Label noise**|Noisy _y_ corrupts empirical risk|Label smoothing, noise-robust losses (e.g., symmetric cross-entropy), co-teaching|
|**Adversarial inputs**|Test inputs crafted to maximize loss|Adversarial training, certified defenses, input preprocessing|
|**Non-IID / correlated data**|Hoeffding-type bounds require independence|Block bootstrap, temporal CV splits, PAC-Bayes bounds for dependent data|
|**Spurious correlations**|ERM exploits shortcuts present in training distribution|Invariant Risk Minimization (IRM), group DRO, causal representation learning|

**Spurious correlations** deserve special attention in research. ERM will always exploit any correlation that reduces training error, regardless of whether it's causal. A model trained on biased data achieves low _ε̂_ but high _ε_ on the true distribution — and the decomposition above makes this visible: approximation error appears low (the model fits training well), but estimation error explodes at test time due to distribution mismatch.
