# Chapter: Data Splits, Models & Cross-Validation

## 1. Bias and Variance Analysis

Understanding the trade-off between bias and variance is critical for diagnosing and improving the performance of learning algorithms.

### 1.1 The Bias-Variance Decomposition

Before describing what bias and variance _look_ like, it helps to see where they come from mathematically. For a learning algorithm producing hypothesis $h$, the expected prediction error at a point $x$ can be decomposed as:

$$ \mathbb{E}\left[(h(x) - y)^2\right] = \underbrace{\left(\mathbb{E}[h(x)] - f(x)\right)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}\left[(h(x) - \mathbb{E}[h(x)])^2\right]}_{\text{Variance}} + \underbrace{\sigma^2_\epsilon}_{\text{Irreducible Noise}} $$

where $f(x)$ is the true underlying function and $\sigma^2_\epsilon$ is the noise inherent in the data.

- **Bias squared** measures how far the average prediction of your model is from the correct value. A high-bias model is systematically wrong.
- **Variance** measures how much your model's predictions fluctuate across different training sets drawn from the same distribution. A high-variance model is inconsistent.
- **Irreducible Noise** is a lower bound on error that cannot be reduced regardless of the algorithm. This sets the Bayes error rate.

The critical insight is that these three terms cannot all be minimized simultaneously. Reducing model complexity reduces variance but increases bias, and vice versa. The goal is to find the sweet spot that minimizes total expected error.

### 1.2 Underfitting (High Bias)

**Definition:** Occurs when a model is too simple to capture the underlying trend of the data.

**Intuition:** The algorithm has strong "preconceptions" (bias) about the data structure — for example, assuming a linear relationship when the true relationship is quadratic.

**Performance:** High error on both training and test sets.

**Diagnosis:** If $J_{\text{train}} \approx J_{\text{dev}}$ and both are high, you have a bias problem.

### 1.3 Overfitting (High Variance)

**Definition:** Occurs when a model is overly complex (e.g., high-degree polynomials) and fits the random noise in the training set rather than the actual trend.

**Intuition:** The model varies significantly depending on the specific random draw of the training data — it memorizes rather than learns.

**Performance:** Extremely low training error but high generalization error.

**Diagnosis:** If $J_{\text{train}}$ is low but $J_{\text{dev}} \gg J_{\text{train}}$, you have a variance problem.

## 2. Regularization

Regularization is one of the most effective techniques to prevent overfitting by penalizing large parameter values.

### 2.1 Optimization Objective

The cost function is modified by adding a regularization term:

$$ J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2 + \frac{\lambda}{2} |\theta|^2 $$

**$\lambda$ (Regularization Parameter):** Controls the trade-off between fitting the data and keeping parameters small.

- If $\lambda = 0$: No regularization (risk of overfitting).
- If $\lambda \to \infty$: Parameters are forced toward zero (risk of underfitting).

**Selecting $\lambda$ in practice:** Train a family of models over a grid, e.g., $\lambda \in {0, 0.001, 0.01, 0.1, 1, 10, 100}$, and select the value that minimizes error on the **dev set**. Never use the test set for this selection.

### 2.2 L2 Regularization (Ridge)

The penalty $\frac{\lambda}{2}|\theta|^2 = \frac{\lambda}{2}\sum_j \theta_j^2$ penalizes all weights continuously. L2 shrinks weights toward zero but rarely sets them exactly to zero. This is the most common form of regularization in neural networks, also called **weight decay**.

### 2.3 L1 Regularization (Lasso)

An important alternative is the **L1 penalty**:

$$ J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2 + \lambda |\theta|_1 $$

where $|\theta|_1 = \sum_j |\theta_j|$.

**Key distinction from L2:** L1 regularization tends to produce **sparse solutions** — it drives many weights _exactly_ to zero. This makes it inherently a feature selection mechanism: irrelevant features are eliminated entirely.

|Property|L2 (Ridge)|L1 (Lasso)|
|---|---|---|
|Solution sparsity|Dense (weights near zero)|Sparse (weights = 0)|
|Sensitivity to outliers|Lower|Higher|
|Closed-form solution|Yes|No (requires iterative solvers)|
|Best when|Most features are relevant|Only a few features matter|


### 2.4 Probabilistic Interpretation: MAP Estimation

Regularization can be viewed as **Maximum A Posteriori (MAP)** estimation, bridging Bayesian statistics and optimization.

In MAP, we seek:

$$ \theta_{\text{MAP}} = \arg\max_\theta ; \log P(\mathcal{D} \mid \theta) + \log P(\theta) $$

The first term is the log-likelihood (fitting the data). The second term is the log-prior over parameters.

- **L2 Regularization corresponds to a Gaussian Prior:** If $P(\theta) \sim \mathcal{N}(0, \tau^2 I)$, the log-prior is $-\frac{1}{2\tau^2}|\theta|^2$. Maximizing this is equivalent to minimizing the L2-penalized cost, with $\lambda = \frac{1}{\tau^2}$.
    
- **L1 Regularization corresponds to a Laplace Prior:** If $P(\theta) \sim \text{Laplace}(0, b)$, the log-prior is $-\frac{1}{b}|\theta|_1$. Maximizing this is equivalent to minimizing the L1-penalized cost. The sharp peak of the Laplace distribution at zero is precisely what encourages sparsity.
    

This probabilistic view is valuable for research engineers: it reframes regularization as _encoding prior beliefs about the model_, opening the door to Bayesian methods and more expressive priors.

## 3. Model Selection & Cross-Validation

To choose hyperparameters (like $\lambda$ or polynomial degree $d$), we must evaluate performance on data the model has not seen during training.

### 3.1 Data Splitting: Train, Dev, and Test Sets

|Split|Purpose|Typical Size (small data)|Typical Size (large data)|
|---|---|---|---|
|**Training Set**|Fit parameters $\theta$|60–70%|98%+|
|**Dev (Validation) Set**|Tune hyperparameters; select models|15–20%|~1%|
|**Test Set**|Final unbiased estimate of generalization|15–20%|~1%|

**Critical rule:** No decisions — architectural or otherwise — should ever be made based on test set results. Using the test set for model selection introduces _optimistic bias_ into your final performance estimate, making the reported number meaningless.

**Data Size Hygiene:** As datasets grow to millions of examples, the proportion needed for dev/test shrinks dramatically — from 30% down to 1% or less. The absolute size of the dev/test set matters more than the proportion.

### 3.2 Cross-Validation Techniques

Cross-validation is used when the dataset is too small to afford a static holdout split, because a single split gives a high-variance estimate of model performance.

#### Simple Holdout

Split the data once (e.g., 70% Train, 30% Dev). Fast and scalable. Best for very large datasets where the holdout is large enough to be statistically reliable.

#### k-Fold Cross-Validation

1. Divide data into $k$ equally-sized folds.
2. For each fold $i = 1, \ldots, k$: train on the remaining $k-1$ folds, evaluate on fold $i$.
3. Average the $k$ error estimates: $\text{CV error} = \frac{1}{k}\sum_{i=1}^{k} \epsilon_i$.

**Why this matters:** Each data point is used for both training and validation exactly once. With $m = 1000$ samples and $k = 10$ folds, each training run uses 900 examples — far more than the 700 a static 70/30 split would provide. The resulting error estimate has **lower variance** than a single holdout split, making it more reliable for hyperparameter selection.

Common choices are $k = 5$ or $k = 10$. Larger $k$ gives lower bias but higher computational cost.

#### Leave-One-Out Cross-Validation (LOOCV)

An extreme case where $k = m$. Each training run uses $m-1$ examples and is evaluated on the single held-out point.

- **Use when:** Dataset is very small ($m < 100$).
- **Advantage:** Maximum use of training data; nearly unbiased error estimate.
- **Disadvantage:** Requires training $m$ separate models; computationally prohibitive for large $m$. The estimates can also exhibit high variance because the $m$ training sets are nearly identical.

|Method|Bias|Variance|Compute Cost|
|---|---|---|---|
|Holdout|Higher|Higher|Low|
|k-Fold (k=10)|Moderate|Moderate|Medium|
|LOOCV|Lowest|Highest|Very High|


## 4. Feature Selection: Forward Search

When dealing with a high-dimensional feature space (e.g., 10,000 words in text classification), we use feature selection to reduce overfitting and improve interpretability.

### 4.1 Forward Search Algorithm

1. Start with an empty feature set $\mathcal{F} = \emptyset$.
2. For each feature $f_j \notin \mathcal{F}$, evaluate model performance on the dev set using $\mathcal{F} \cup {f_j}$.
3. Add the single feature that most improves dev set performance: $\mathcal{F} \leftarrow \mathcal{F} \cup {f^*}$.
4. Repeat until adding features no longer provides significant improvement.

### 4.2 Computational Cost and Alternatives

**The cost problem:** With $n = 10{,}000$ features, the first step alone requires training approximately 10,000 separate models. Total complexity is $O(n^2)$ model evaluations — often infeasible.

**Cheaper alternatives:**

- **Filter Methods (Mutual Information):** Score each feature $f_j$ independently by $I(f_j; y)$ before any model is trained. Features are ranked and a threshold is applied. This is $O(n)$ and scales to very large feature spaces, but ignores interactions between features.
- **Embedded Methods (L1 Regularization):** Let the model itself perform feature selection during training via an L1 penalty. Efficient and accounts for feature interactions within the model family.
- **Recursive Feature Elimination (RFE):** Train a model, rank features by coefficient magnitude, remove the weakest, and repeat.

In practice, L1 regularization is the most common choice in modern ML pipelines because it performs feature selection implicitly with no additional computational overhead.

## 5. Meta-Insights

### 5.1 The Bias-Variance Lens on Model Development

Every design decision in ML — adding layers, changing optimizers, collecting data — can be interpreted through the bias-variance lens. Before making a change, ask: _is my problem primarily bias or variance?_

|Problem|Remedies|
|---|---|
|High Bias|Larger model, more features, less regularization, different architecture|
|High Variance|More training data, stronger regularization, feature selection, dropout, early stopping|
|Both|More data often helps both; architecture redesign may be needed|

### 5.2 The Generalization Goal

The ultimate aim of an ML engineer is **not** to explain training data perfectly, but to build a model that resists noise and generalizes to unseen distributions. Memorization is cheap; generalization is hard. A model's value is measured entirely on data it was never trained on — ideally data that looks like what it will encounter in deployment.

## Summary of Key Formulas

|Concept|Formula|
|---|---|
|Bias-Variance Decomposition|$\text{Error} = \text{Bias}^2 + \text{Variance} + \sigma^2_\epsilon$|
|L2-Regularized Cost|$J(\theta) = \frac{1}{2m}\sum(h_\theta(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2}\|\theta\|^2$|
|L1-Regularized Cost|$J(\theta) = \frac{1}{2m}\sum(h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda\|\theta\|_1$|
|MAP Objective|$\theta_{\text{MAP}} = \arg\max_\theta \log P(\mathcal{D}\mid\theta) + \log P(\theta)$|
|k-Fold CV Error|$\text{CV} = \frac{1}{k}\sum_{i=1}^{k} \epsilon_i$|