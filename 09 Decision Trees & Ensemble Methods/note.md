# Chapter: Decision Trees and Ensemble Methods

## 1. Decision Trees

### 1.1 Motivation and Recursive Partitioning

Decision trees represent a departure from classical linear models (like Logistic Regression or SVMs) toward non-linear classification and regression. They partition the input space into a set of rectangular regions.

The algorithm uses **Recursive Binary Partitioning**:

- **Top-down:** Starts with the entire dataset as the root.
- **Greedy:** At each node, it selects the split that minimizes a specific loss function at that step, without looking ahead to future splits.

> **[Extension] ERM Framework for Decision Trees**
>
> Formally, a decision tree defines a hypothesis class $\mathcal{H}_T$ consisting of all piecewise-constant functions over rectangular partitions of the input space. Learning a tree is then an **Empirical Risk Minimization** problem:
>
> $$\hat{h} = \arg\min_{h \in \mathcal{H}_T} \frac{1}{n} \sum_{i=1}^{n} \mathcal{L}(h(x_i), y_i)$$
>
> However, finding the globally optimal tree is **NP-hard** (the number of possible binary partitions grows exponentially with depth). Greedy recursive splitting is therefore a **tractable approximation** — it minimizes risk locally at each node rather than globally over the full tree structure. This is why a greedy-grown tree is not guaranteed to be the globally optimal tree, even on training data.
>
> **Local vs. Global Hypothesis Class:** Each leaf of the tree corresponds to a *local* hypothesis (a constant prediction within a region), while the full tree defines the *global* hypothesis. The expressiveness of $\mathcal{H}_T$ grows with depth, but so does the risk of overfitting — the global hypothesis class becomes too large relative to the training data.


### 1.2 Defining the Split

A split at a region $R_p$ is defined by a feature $j$ and a threshold $t$:

$$R_1 = \{x \in R_p \mid x_j < t\}, \quad R_2 = \{x \in R_p \mid x_j \geq t\}$$

The goal is to choose $(j, t)$ to maximize the "purity" of the resulting children regions.

### 1.3 Splitting Criteria (Loss Functions)

Let $\hat{p}_{mc}$ be the proportion of training observations in region $m$ that belong to class $c$.

1. **Misclassification Error:** $1 - \max_c(\hat{p}_{mc})$. Often too insensitive for tree growth.

2. **Cross-Entropy (Information Gain):**

$$\mathcal{L}_{cross} = -\sum_{c=1}^{C} \hat{p}_{mc} \log_2 \hat{p}_{mc}$$

3. **Gini Impurity:**

$$\mathcal{L}_{Gini} = \sum_{c=1}^{C} \hat{p}_{mc}(1 - \hat{p}_{mc})$$

> **[Extension] Deep Dive: Why use Entropy/Gini over Misclassification? (Corrected Argument)**
>
> The correct explanation is not about gradients — greedy splitting is a **combinatorial search**, not a continuous optimization, so gradient-based arguments are a category error here.
>
> The key property is **strict concavity**. Misclassification error has the form $1 - \max_c(p_c)$, which is **piecewise linear** and not strictly concave. This means there exist splits that meaningfully change the class distribution in the children yet produce **no decrease in misclassification loss** — because both parent and children fall on the same linear segment.
>
> Entropy and Gini, being strictly concave, guarantee that **any split that changes the class distribution will strictly decrease the weighted child loss** relative to the parent. This property is called **sensitivity to distributional change**, not gradient sensitivity.
>
> Formally, by Jensen's inequality applied to a strictly concave $\phi$:
>
> $$\phi\!\left(\frac{n_1}{n}p_1 + \frac{n_2}{n}p_2\right) > \frac{n_1}{n}\phi(p_1) + \frac{n_2}{n}\phi(p_2)$$
>
> ensuring the parent impurity always exceeds the weighted average child impurity after any non-trivial split. Misclassification error does **not** satisfy this for all splits.
>
> Additionally, Entropy corresponds to minimizing KL divergence between the post-split and pre-split class distributions, connecting it formally to information theory:
>
> $$\text{Information Gain} = D_{KL}(p_{\text{child}} \,\|\, p_{\text{parent}})$$


### 1.4 Regularization and Pruning

Unconstrained trees are prone to overfitting (high variance). Strategies to regularize include:

- **Minimum leaf size:** Stop splitting if a node has fewer than $n$ examples.
- **Maximum depth:** Limit the number of questions asked.
- **Pruning:** Grow a full tree and then collapse nodes that do not significantly improve performance on a validation set.

> **[Extension] Why Do Trees Have High Variance? (Instability Analysis)**
>
> The high variance of decision trees is not simply a consequence of overfitting in the generic sense. It arises from the **instability of greedy splits**: a small perturbation in the training data (e.g., removing or modifying a few points near a split threshold) can cause an entirely different feature to be selected at the root, cascading into a completely different tree structure. This is in contrast to linear models, where small data changes produce small coefficient changes.
>
> Formally, let $T(S)$ denote the tree trained on dataset $S$. Decision trees satisfy:
>
> $$\mathbb{V}[T(S)] \gg \mathbb{V}[\text{linear model trained on } S]$$
>
> for comparable complexity, because the hypothesis selected at each node is **discontinuously sensitive** to the data. This is precisely why ensemble methods (Bagging, Random Forests) are so effective: they reduce variance by averaging over this instability.

> **[Extension] Cost-Complexity Pruning (CART)**
>
> The standard pruning mechanism in CART introduces a regularized objective:
>
> $$R_\alpha(T) = R(T) + \alpha |T|$$
>
> where $R(T)$ is the training error, $|T|$ is the number of leaves, and $\alpha \geq 0$ is a complexity parameter tuned by cross-validation. For each $\alpha$, there exists a unique smallest subtree minimizing $R_\alpha(T)$. As $\alpha$ increases, the pruned tree grows simpler — this is the continuous equivalent of the depth/leaf-size constraints, and is directly analogous to L1 regularization penalizing model complexity.

## 2. Ensemble Methods

### 2.1 Theoretical Foundation: Bias-Variance

If we have $n$ i.i.d. random variables $X_i$ with variance $\sigma^2$, the variance of their mean is $\frac{\sigma^2}{n}$. Ensemble methods aim to reduce total error by combining multiple "weak" learners.

For correlated variables with correlation $\rho$, the variance of the mean is:

$$\text{Var}(\bar{X}) = \rho\sigma^2 + \frac{1-\rho}{n}\sigma^2$$

To reduce variance effectively, we must both increase $n$ (number of models) and decrease $\rho$ (de-correlate models).


### 2.2 Bagging (Bootstrap Aggregation)

Bagging reduces variance by training separate models on different samples of the data.

1. **Bootstrapping:** Generate $M$ new datasets by sampling with replacement from the original training set $S$.
2. **Aggregation:** For regression, average the predictions; for classification, use majority voting.

$$f_{\text{bag}}(x) = \frac{1}{M} \sum_{m=1}^{M} \hat{f}^*_m(x)$$

> **[Extension] Out-of-Bag (OOB) Error Estimation**
>
> Each bootstrap sample leaves out approximately $1/e \approx 36.8\%$ of training examples. These **out-of-bag** samples can be used to estimate generalization error without a separate validation set: each training point is predicted only by the trees for which it was not used in training. This gives an unbiased estimate of test error essentially "for free," and is one of the practical advantages of Bagging/Random Forests.


### 2.3 Random Forests

Random Forests improve upon Bagging by further de-correlating the trees. At each split, the algorithm considers only a random subset of features (typically $\sqrt{F}$). This prevents a single dominant feature from making all trees look identical, thereby driving down $\rho$.

> **[Extension] Feature Importance**
>
> Random Forests provide two natural measures of feature importance:
>
> - **Impurity-based importance:** Sum of impurity reduction at all nodes where a feature is used, weighted by the number of samples. Efficient but can be biased toward high-cardinality features.
> - **Permutation importance:** Measure the decrease in OOB accuracy when the values of a feature are randomly shuffled. More robust and model-agnostic.
>
> These are among the primary reasons tree-based ensembles remain preferred over neural networks for tabular data in many research settings — they offer interpretability alongside strong predictive performance.


### 2.4 Boosting (AdaBoost and the General Framework)

Unlike Bagging, Boosting is an additive process focused on reducing bias. Models are trained sequentially, with each subsequent model focusing more on the examples that previous models misclassified by increasing their weights.

The final output is a weighted sum of all models:

$$G(x) = \text{sign}\!\left(\sum_{m=1}^{M} \alpha_m G_m(x)\right)$$

where $\alpha_m$ is the weight assigned to the $m$-th classifier based on its accuracy.

> **[Extension] Boosting Under the Objective Function Lens**
>
> AdaBoost is a special case of a much more general framework: **Gradient Boosting**. The unifying perspective is to view the ensemble as a functional optimization problem. At each step $m$, we fit a new weak learner $h_m$ to the **negative gradient** of a loss function $\mathcal{L}$ with respect to the current model prediction $F_{m-1}(x)$:
>
> $$h_m = \arg\min_h \sum_{i=1}^{n} \left[-\frac{\partial \mathcal{L}(y_i, F(x_i))}{\partial F(x_i)}\bigg|_{F=F_{m-1}}\right] - h(x_i))^2$$
>
> Different loss functions yield different boosting algorithms:
>
> | Loss Function | Algorithm |
> |---|---|
> | Exponential: $e^{-yF(x)}$ | AdaBoost |
> | Log-loss: $\log(1 + e^{-yF(x)})$ | LogitBoost |
> | Squared error: $(y - F(x))^2$ | L2 Boosting |
> | Arbitrary differentiable $\mathcal{L}$ | Gradient Boosting (Friedman, 2001) |
>
> **XGBoost** extends this framework by adding (1) a regularization term on the leaf weights to the objective, and (2) a **second-order Taylor expansion** of the loss for more accurate gradient approximation:
>
> $$\mathcal{L}^{(m)} \approx \sum_{i} \left[g_i h_m(x_i) + \frac{1}{2} k_i h_m(x_i)^2\right] + \Omega(h_m)$$
>
> where $g_i$ and $k_i$ are the first and second derivatives of the loss. This perspective shows that AdaBoost is not an isolated algorithm but one instance of a continuum — choosing the loss function is equivalent to specifying what kind of "errors" the model should focus on.

## 3. Meta-Insights & Comparison

| Feature | Single Decision Tree | Random Forest (Bagging) | Boosting (AdaBoost/XGBoost) |
|---|---|---|---|
| Main Goal | Interpretability | Variance Reduction | Bias Reduction |
| Structure | Standalone | Parallel (Independent) | Sequential (Dependent) |
| Complexity | Low | Medium | High |
| Sensitivity | High (to outliers) | Robust (averaging) | High (can overfit noise) |
| Optimization View | Greedy ERM approx. | Avg. over unstable hypotheses | Functional gradient descent |

**Conclusion:** Decision trees are rarely used in isolation due to their high variance — a consequence of the instability of greedy splits. However, when combined into ensembles like Random Forests or Gradient Boosted Trees, they become some of the most powerful and widely used algorithms in competitive machine learning and applied research.

From a theoretical standpoint, the progression from single trees → Bagging → Random Forests → Gradient Boosting reflects a principled journey along the bias-variance tradeoff: each step introduces a mechanism to either reduce variance (de-correlation, averaging) or reduce bias (sequential residual fitting), grounded in a coherent ERM and functional optimization framework.