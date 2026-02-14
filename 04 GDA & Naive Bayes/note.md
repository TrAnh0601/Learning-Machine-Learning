# Chapter: Generative Learning Algorithms


## 1. Introduction: Discriminative vs. Generative Models

### Discriminative Learning Algorithms

Algorithms like Logistic Regression or Linear Regression try to find a decision boundary (e.g., a straight line) that separates classes (positive and negative examples) directly.

They model $P(y|x)$ directly or learn a mapping from $x$ to $y$.

**Geometric Intuition:** Discriminative algorithms draw a hyperplane through the data that best separates positive from negative examples. They focus solely on finding the optimal boundary, without modeling what each class looks like individually.

### Generative Learning Algorithms

Instead of searching for a separator, these algorithms build a model of what each class "looks like" individually.

For example, in cancer classification, it builds a model for malignant tumors and a separate model for benign tumors.

They learn:

- $P(x|y)$: Features given the class
- $P(y)$: Class prior

**Geometric Intuition:** Rather than finding a boundary, generative models build probability distributions around each class's examples. Classification is done by determining which distribution a new example is more likely to have been generated from.

### Classification via Bayes' Rule

When a new example arrives, we compare it against both models to see which one it matches best:

$$P(y = 1|x) = \frac{P(x|y = 1)P(y = 1)}{P(x)}$$

where the denominator is:

$$P(x) = P(x|y = 0)P(y = 0) + P(x|y = 1)P(y = 1)$$


## 2. Gaussian Discriminant Analysis (GDA)

GDA is a generative algorithm used when input features are continuous ($x \in \mathbb{R}^n$).

### 2.1 Model Assumptions

We assume the likelihood of features given the class follows a Multivariate Gaussian Distribution:

$$y \sim \text{Bernoulli}(\phi)$$

$$x|y = 0 \sim \mathcal{N}(\mu_0, \Sigma)$$

$$x|y = 1 \sim \mathcal{N}(\mu_1, \Sigma)$$

**Crucial Assumption:** Both classes share the same covariance matrix $\Sigma$.

**Why This Matters:** The shared covariance matrix is what makes GDA produce a linear decision boundary. The decision boundary is where $P(y=0|x) = P(y=1|x)$. When we compute this using the Gaussian density formula, the quadratic terms $x^T\Sigma^{-1}x$ appear in both class densities with identical coefficients. Since $\Sigma$ is the same for both classes, these quadratic terms cancel exactly when we compute the log-odds ratio, leaving only linear terms in $x$.

**Geometric Visualization:** Picture two Gaussian clouds (ellipsoids) in feature space, centered at $\mu_0$ and $\mu_1$. Because they share the same covariance matrix $\Sigma$, both ellipsoids have identical shape and orientation—only their centers differ. The decision boundary is the perpendicular bisector of the line segment connecting $\mu_0$ to $\mu_1$, adjusted for the ellipsoid shape and class priors.

### 2.2 Maximum Likelihood Estimation (MLE)

To train the model, we maximize the joint log-likelihood of the data.

#### Derivation of Parameters

The log-likelihood is given by:

$$\ell(\phi, \mu_0, \mu_1, \Sigma) = \log \prod_{i=1}^{m} P(x^{(i)}, y^{(i)}) = \sum_{i=1}^{m} \log P(x^{(i)}|y^{(i)}) + \sum_{i=1}^{m} \log P(y^{(i)})$$

To find the parameters, we take the partial derivatives with respect to each parameter ($\phi, \mu_0, \mu_1, \Sigma$) and set them to zero:

**1. For $\phi$:** Differentiating with respect to $\phi$ yields the fraction of positive examples.

$$\phi = \frac{1}{m} \sum_{i=1}^{m} \mathbb{1}{y^{(i)} = 1}$$

**2. For $\mu_0$ and $\mu_1$:** Differentiating the Gaussian log-density with respect to $\mu$ yields the weighted average of $x$'s for that class.

$$\mu_0 = \frac{\sum_{i=1}^{m} \mathbb{1}{y^{(i)} = 0} x^{(i)}}{\sum_{i=1}^{m} \mathbb{1}{y^{(i)} = 0}}$$

$$\mu_1 = \frac{\sum_{i=1}^{m} \mathbb{1}{y^{(i)} = 1} x^{(i)}}{\sum_{i=1}^{m} \mathbb{1}{y^{(i)} = 1}}$$

**3. For $\Sigma$:** Maximizing with respect to the covariance matrix yields the empirical covariance.

$$\Sigma = \frac{1}{m} \sum_{i=1}^{m} (x^{(i)} - \mu_{y^{(i)}})(x^{(i)} - \mu_{y^{(i)}})^T$$

**Interpretation:** These formulas are intuitive: $\phi$ is the fraction of positive training examples; $\mu_0$ and $\mu_1$ are the centroids (centers of mass) of each class; $\Sigma$ measures the spread of data around these centroids. The MLE solution is exactly what you would compute manually if asked to describe each class's distribution.


## 3. GDA and Logistic Regression

### 3.1 Theorem: GDA Implies Logistic Regression

If $P(x|y)$ is multivariate Gaussian with a shared covariance matrix $\Sigma$, then the posterior $P(y = 1|x)$ follows a Logistic function.

#### Proof

Assumptions: $$x|y = 0 \sim \mathcal{N}(\mu_0, \Sigma)$$ $$x|y = 1 \sim \mathcal{N}(\mu_1, \Sigma)$$

We start with Bayes' rule:

$$P(y = 1|x) = \frac{P(x|y = 1)P(y = 1)}{P(x|y = 1)P(y = 1) + P(x|y = 0)P(y = 0)}$$

Divide numerator and denominator by the numerator's terms to get the form:

$$P(y = 1|x) = \frac{1}{1 + \frac{P(x|y=0)P(y=0)}{P(x|y=1)P(y=1)}}$$

We examine the log-ratio in the denominator exponent (the "log-odds"):

$$\log \frac{P(x|y = 1)P(y = 1)}{P(x|y = 0)P(y = 0)} = \log \frac{P(y = 1)}{P(y = 0)} + \log \frac{P(x|y = 1)}{P(x|y = 0)}$$

Substituting the Gaussian densities (the normalization constant $C = (2\pi)^{-n/2}|\Sigma|^{-1/2}$ cancels out):

$$= \text{const} + \left[-\frac{1}{2}(x - \mu_1)^T\Sigma^{-1}(x - \mu_1)\right] - \left[-\frac{1}{2}(x - \mu_0)^T\Sigma^{-1}(x - \mu_0)\right]$$

Expanding the quadratic term:

$$(x - \mu)^T\Sigma^{-1}(x - \mu) = x^T\Sigma^{-1}x - 2\mu^T\Sigma^{-1}x + \mu^T\Sigma^{-1}\mu$$

**Key Insight:** The quadratic term $x^T\Sigma^{-1}x$ appears in both class densities with the same sign (because $\Sigma$ is shared). They cancel out.

We are left with terms linear in $x$:

$$= x^T\Sigma^{-1}(\mu_1 - \mu_0) + \text{constants}$$

This is of the form $\theta^T x + \theta_0$. Thus:

$$P(y = 1|x) = \frac{1}{1 + e^{-(\theta^T x + \theta_0)}}$$

This confirms GDA implies a Logistic Regression form.

### 3.2 Comparison: GDA vs. Logistic Regression

|Feature|GDA (Generative)|Logistic Regression (Discriminative)|
|---|---|---|
|**Assumptions**|Strong: $x \mid y$ must be Gaussian with shared $\Sigma$|Weak: only assumes linear decision boundary|
|**Data Efficiency**|High: Asymptotically efficient if data is truly Gaussian. Requires less data to converge.|Lower: Needs more data to reach the same performance level if data is Gaussian|
|**Robustness**|Brittle: If Gaussian assumption violated (e.g., Poisson, heavy tails), converges to biased solution|Consistent: More robust. Even if $x \mid y$ not Gaussian, finds correct boundary|
|**Parameters**|$O(n^2)$: needs to estimate full covariance matrix $\Sigma$|$O(n)$: only learns weight vector $\theta$|
|**Relationship**|Implies Logistic Regression|Does NOT imply GDA|

**Decision Rule for Practice:**

Use GDA when:

- You are confident the data is approximately Gaussian
- You have limited training data
- You need probabilistic interpretations

Use Logistic Regression when:

- You are unsure about the underlying distribution
- You have plenty of training data
- You want robustness over efficiency

### 3.3 Concrete Example: Medical Diagnosis

**Scenario:** Predicting diabetes from patient measurements (glucose, BMI, blood pressure).

**When GDA wins:**

- Small hospital with only 200 patients
- Medical measurements are often normally distributed
- GDA learns from limited data more effectively, achieving 85% accuracy while Logistic Regression plateaus at 78%

**When Logistic Regression wins:**

- Large dataset (10,000 patients) including outliers and measurement errors
- Some features (e.g., number of pregnancies) are discrete, violating Gaussian assumption
- Logistic Regression achieves 92% accuracy while GDA's rigid assumptions cause it to plateau at 88%


## 4. Naive Bayes

When input features are discrete (e.g., text classification where $x_j \in {0, 1}$), we use Naive Bayes.

### 4.1 The Naive Assumption

We assume features $x_1, \ldots, x_n$ are conditionally independent given the class $y$.

$$P(x_1, \ldots, x_n|y) = \prod_{j=1}^{n} P(x_j|y)$$

**Why "Naive"?** This assumption is almost always violated in practice. For example, in spam emails, the words "Nigerian" and "Prince" are highly correlated—if one appears, the other is more likely. Naive Bayes treats them as independent.

**Why It Works Anyway:** Despite the violated independence assumption, the ranking of class probabilities often remains correct. Even if the exact probability values are wrong, we only care about which class has higher probability. Additionally, the independence assumption dramatically reduces the number of parameters from exponential to linear in the number of features.

Parameter reduction:

- Without independence: $2^n$ parameters for binary features
- With independence: $n$ parameters

### 4.2 The Model (Bernoulli Naive Bayes)

For a spam filter where $y = 1$ is spam:

**Parameters:**

- $\phi_{j|y=1} = P(x_j = 1|y = 1)$: Probability word $j$ appears in spam
- $\phi_{j|y=0} = P(x_j = 1|y = 0)$: Probability word $j$ appears in non-spam
- $\phi_y = P(y = 1)$: Prior probability of spam

**Maximum Likelihood Estimates:**

$$\phi_{j|y=1} = \frac{\text{count}(x_j = 1, y = 1)}{\text{count}(y = 1)}$$

$$\phi_{j|y=0} = \frac{\text{count}(x_j = 1, y = 0)}{\text{count}(y = 0)}$$

$$\phi_y = \frac{\text{count}(y = 1)}{m}$$

#### Naive Bayes as a Linear Classifier

While generative, Naive Bayes behaves like a linear classifier in the log-feature space.

Consider the log-odds ratio:

$$\log \frac{P(y = 1|x)}{P(y = 0|x)} = \log \frac{P(y = 1)}{P(y = 0)} + \sum_{j=1}^{n} \log \frac{P(x_j|y = 1)}{P(x_j|y = 0)}$$

For binary features $x_j \in {0, 1}$, let $p_j = P(x_j = 1|y = 1)$ and $q_j = P(x_j = 1|y = 0)$. The term inside the sum contributes a weight $w_j = \log \frac{p_j}{q_j}$ if $x_j = 1$, and a different constant if $x_j = 0$.

Thus, the decision rule can be rewritten as:

$$\theta^T x + \theta_0 > 0$$

This proves that the decision boundary of Bernoulli Naive Bayes is linear in $x$.

**Geometric Interpretation:** Each feature $j$ contributes a weighted vote: $w_j = \log(p_j/q_j)$. If word $j$ appears much more frequently in spam ($p_j \gg q_j$), it receives a large positive weight. If it appears more in legitimate emails, it receives a negative weight. The final decision is a weighted sum—exactly like logistic regression, but with weights determined by probability ratios.

### 4.3 Laplace Smoothing

**Problem:** If a word $x_j$ never appears in the training set for a class (e.g., $y = 1$), then $P(x_j|y = 1) = 0$. This makes the entire product probability 0, regardless of all other words.

**Solution:** Add a small "pseudocount" (usually 1) to the numerator and $k$ (number of values) to the denominator.

$$\phi_{j|y=1} = \frac{\sum_{i=1}^{m} \mathbb{1}{x_j^{(i)} = 1, y^{(i)} = 1} + 1}{\sum_{i=1}^{m} \mathbb{1}{y^{(i)} = 1} + 2}$$

**Why +2 in the denominator?** For binary features, there are $k = 2$ possible values (0 or 1), so we add $k$ to the denominator to maintain probabilistic consistency.

**General Formula:** For features with $k$ possible values:

$$\phi_{j|y=c} = \frac{\text{count}(x_j = \text{value}, y = c) + 1}{\text{count}(y = c) + k}$$

**Intuition:** Laplace smoothing is equivalent to starting with a "virtual" dataset where each word/class combination has already been observed once. It represents a Bayesian prior belief that every word is possible, even if not yet observed. The +1 is negligible when counts are large but crucial when counts are zero.


## 5. Deep Dive Insights

### 5.1 GDA vs. QDA (Quadratic Discriminant Analysis)

**Linear Decision Boundary (GDA):** By assuming $\Sigma_0 = \Sigma_1 = \Sigma$, the quadratic terms cancel, resulting in a linear boundary.

**Quadratic Decision Boundary (QDA):** If we allow separate covariance matrices $\Sigma_0 \neq \Sigma_1$, the quadratic terms do not cancel. The boundary becomes a conic section (ellipse, parabola, hyperbola). This is more flexible but has higher variance (prone to overfitting).

**Decision boundary for QDA:**

$$x^T(\Sigma_1^{-1} - \Sigma_0^{-1})x + \text{linear terms} + \text{constant} = 0$$

This is quadratic in $x$.

**When to use each:**

- GDA (Linear): Classes have similar spread, limited data, or need interpretability
- QDA (Quadratic): Classes have clearly different shapes/spreads, plenty of data, need flexibility

### 5.2 Asymptotic Efficiency vs. Consistency

**GDA (Asymptotically Efficient):** If the data truly comes from a Gaussian distribution, GDA is asymptotically efficient—meaning no other algorithm can learn the parameters better with fewer samples. It achieves the Cramér-Rao lower bound on variance and converges to the true parameters at rate $O(1/\sqrt{m})$.

**Logistic Regression (Consistent):** Logistic Regression is consistent for a broader class of problems. If the true decision boundary is linear, Logistic Regression will eventually find it as $m \to \infty$, regardless of the underlying distribution of $x|y$. However, GDA is model-misspecified if the data is not Gaussian; it will converge to specific parameters, but those parameters will define a biased decision boundary that is not optimal.

**Mathematical formulation:**

Let $\theta_{\text{GDA}}$ and $\theta_{\text{LR}}$ be the limiting parameters as $m \to \infty$.

- If data is Gaussian: $\theta_{\text{GDA}} = \theta_{\text{true}}$ (unbiased), and GDA converges faster
- If data is non-Gaussian but boundary is linear: $\theta_{\text{LR}} = \theta_{\text{true}}$ (consistent), but $\theta^*_{\text{GDA}} \neq \theta^*_{\text{true}}$ (biased)

**The Tradeoff:** GDA bets heavily on the Gaussian assumption. If correct, it wins with fewer samples. If wrong, it converges to the wrong answer. Logistic Regression is more conservative—it makes weaker assumptions and is therefore more robust.

### 5.3 Extensions of Naive Bayes

**Multinomial Naive Bayes:** For count data (e.g., word counts in documents):

$$P(x|y) = \frac{(\sum_j x_j)!}{\prod_j x_j!} \prod_{j=1}^{n} p_{j|y}^{x_j}$$

where $x_j$ is the count of word $j$ in the document.

Use case: Text classification where word frequency matters (sentiment analysis, topic classification).

**Gaussian Naive Bayes:** For continuous features assumed to be Gaussian:

$$P(x_j|y) = \frac{1}{\sqrt{2\pi\sigma_{j|y}^2}} \exp\left(-\frac{(x_j - \mu_{j|y})^2}{2\sigma_{j|y}^2}\right)$$

Difference from GDA: Assumes features are independent given $y$. GDA models the full covariance $\Sigma$.

Parameter reduction:

- GDA: $O(n^2)$ parameters (full covariance matrix)
- Gaussian Naive Bayes: $O(n)$ parameters (diagonal covariance only)

### 5.4 When Naive Bayes Outperforms Despite Violations

Research has shown that Naive Bayes can outperform more sophisticated methods when:

1. High-dimensional sparse data (e.g., text classification with 10,000+ features)
2. Strong class discrimination (even though correlations exist, class-conditional distributions are very different)
3. Limited training data (the strong independence assumption acts as regularization)
4. Robust ranking (we only need correct ordering of $P(y|x)$, not exact values)