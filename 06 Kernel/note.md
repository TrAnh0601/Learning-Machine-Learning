# Chapter: Naive Bayes Extensions & Support Vector Machines (SVM)

## 1. Introduction

This chapter concludes the discussion on generative learning algorithms, specifically focusing on practical enhancements to Naive Bayes, and introduces the foundational concepts of Support Vector Machines (SVMs). We explore how to handle rare events in discrete distributions and establish the geometric principles behind high-margin classifiers.

## 2. Naive Bayes Enhancements

### 2.1 The Problem of Zero Probabilities

In the standard Naive Bayes model, parameters are estimated using Maximum Likelihood Estimation (MLE). If a feature (e.g., a specific word "NIPS") never appears in the training data for a class $y = 1$, the estimated probability is:

$$\phi_{j|y=1} = \frac{\sum_{i=1}^m 1{x_j^{(i)} = 1, y^{(i)} = 1}}{\sum_{i=1}^m 1{y^{(i)} = 1}} = 0$$

During inference, if a new email contains this word, the product $\prod P(x_j|y)$ becomes zero for both classes, leading to a $\frac{0}{0}$ indeterminate form in Bayes' rule.

### 2.2 Laplace Smoothing

To address this, we apply **Laplace Smoothing**, where we "pretend" to have seen each possible outcome at least once by adding 1 to the numerator and $k$ (the number of possible values) to the denominator. For a multinomial random variable $x \in {1,\ldots, k}$:

$$\phi_j = \frac{\sum_{i=1}^m 1{x^{(i)} = j} + 1}{m + k}$$

**Insight:** This prevents the model from assigning zero probability to events simply because they were not observed in a finite training set.

## 3. Text Classification: Multinomial Event Model

In the Multinomial Event Model, an email is represented as a vector of indices $(x_1, x_2,\ldots, x_n)$, where $n$ is the email length and each $x_j \in {1,\ldots, |V|}$.

- **Multivariate Bernoulli Model:** Tracks only the presence/absence of words (binary ${0, 1}$).
- **Multinomial Event Model:** Tracks which word appears at each position, inherently accounting for word frequency.

## 4. Support Vector Machines (SVM)

### 4.1 Notation and Setup

For SVMs, we use the following notation:

- **Labels:** $y \in {-1, 1}$ (instead of ${0, 1}$).
- **Hypothesis:** $h_{w,b}(x) = g(w^T x + b)$, where $g(z) = 1$ if $z \geq 0$ and $-1$ otherwise.
- **Parameters:** Weight vector $w$ and intercept term $b$ (bias).

### 4.2 Functional Margin

The **functional margin** of a training example $(x^{(i)}, y^{(i)})$ is:

$$\hat{\gamma}^{(i)} = y^{(i)}(w^T x^{(i)} + b)$$

This measures how confident and correct our prediction is:

- If $y^{(i)} = 1$ and $w^T x^{(i)} + b \gg 0$, then $\hat{\gamma}^{(i)}$ is large and positive → confident correct prediction
- If $y^{(i)} = -1$ and $w^T x^{(i)} + b \ll 0$, then $\hat{\gamma}^{(i)}$ is large and positive → confident correct prediction
- If $\hat{\gamma}^{(i)} > 0$, the prediction is correct
- If $\hat{\gamma}^{(i)} < 0$, the prediction is wrong

**Problem:** We can make $\hat{\gamma}^{(i)}$ arbitrarily large by scaling $w$ and $b$ (e.g., replacing $w$ with $2w$ and $b$ with $2b$).

### 4.3 Geometric Margin

The **geometric margin** is the actual Euclidean distance from point $x^{(i)}$ to the decision boundary:

$$\gamma^{(i)} = \frac{y^{(i)}(w^T x^{(i)} + b)}{|w|}$$

The geometric margin is independent of scaling because both the numerator and denominator scale proportionally.

## 5. The Optimal Margin Classifier

### 5.1 Goal

We want to find the decision boundary that maximizes the margin (distance) to the nearest training examples. This gives us the most "confident" separator.

### 5.2 Optimization Problem

To maximize the geometric margin $\gamma$, we can fix the functional margin $\hat{\gamma} = 1$ and minimize $|w|$:

$$\min_{w,b} \frac{1}{2}|w|^2$$

$$\text{subject to } y^{(i)}(w^T x^{(i)} + b) \geq 1, \quad i = 1,\ldots, m$$

**Why $\frac{1}{2}|w|^2$?**

- Minimizing $|w|$ is equivalent to maximizing $\frac{1}{|w|}$ (the margin)
- We use $\frac{1}{2}|w|^2$ for mathematical convenience (easier derivatives)

### 5.3 Support Vectors

The training examples that lie exactly on the margin (where $y^{(i)}(w^T x^{(i)} + b) = 1$) are called **support vectors**. These are the only points that matter for determining the decision boundary.

Points that are far from the boundary ($y^{(i)}(w^T x^{(i)} + b) > 1$) don't affect the solution at all.

## 6. Lagrangian Dual Formulation

### 6.1 Why the Dual?

The dual formulation allows us to:

1. Solve the optimization problem more efficiently
2. Introduce the kernel trick later (for non-linear boundaries)

### 6.2 The Lagrangian

We introduce Lagrange multipliers $\alpha_i \geq 0$ for each constraint:

$$\mathcal{L}(w, b, \alpha) = \frac{1}{2}|w|^2 - \sum_{i=1}^m \alpha_i [y^{(i)}(w^T x^{(i)} + b) - 1]$$

### 6.3 Dual Problem

Taking derivatives and setting them to zero gives us:

$$w = \sum_{i=1}^m \alpha_i y^{(i)} x^{(i)}$$

$$\sum_{i=1}^m \alpha_i y^{(i)} = 0$$

The dual optimization problem becomes:

$$\max_{\alpha} \sum_{i=1}^m \alpha_i - \frac{1}{2}\sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y^{(i)} y^{(j)} \langle x^{(i)}, x^{(j)} \rangle$$

$$\text{subject to } \alpha_i \geq 0, \quad i = 1,\ldots, m$$

$$\sum_{i=1}^m \alpha_i y^{(i)} = 0$$

**Key observation:** The solution only depends on **inner products** $\langle x^{(i)}, x^{(j)} \rangle$ between training examples.

### 6.4 Making Predictions

Once we solve for $\alpha^*$, predictions for new points are:

$$h(x) = g\left(\sum_{i=1}^m \alpha_i^* y^{(i)} \langle x^{(i)}, x \rangle + b^*\right)$$

Most $\alpha_i$ will be zero—only the support vectors have $\alpha_i > 0$.

## 7. Soft-Margin SVM

### 7.1 The Problem with Hard Margins

The optimization problem we've seen so far assumes the data is **linearly separable**. Problems:

- Real data often has noise or outliers
- A single mislabeled point can make the problem unsolvable
- We might overfit to noisy data

### 7.2 Slack Variables

We introduce **slack variables** $\xi_i \geq 0$ that allow some points to violate the margin:

$$y^{(i)}(w^T x^{(i)} + b) \geq 1 - \xi_i$$

**Interpretation:**

- $\xi_i = 0$: Point is outside the margin (correctly classified)
- $0 < \xi_i < 1$: Point is inside the margin but still correctly classified
- $\xi_i \geq 1$: Point is misclassified

### 7.3 New Optimization Problem

$$\min_{w,b,\xi} \frac{1}{2}|w|^2 + C\sum_{i=1}^m \xi_i$$

$$\text{subject to } y^{(i)}(w^T x^{(i)} + b) \geq 1 - \xi_i$$

$$\xi_i \geq 0, \quad i = 1,\ldots, m$$

**Parameter $C$:** Controls the trade-off

- **Large $C$:** Penalize violations heavily → narrow margin, fewer errors
- **Small $C$:** Allow more violations → wider margin, more robust

### 7.4 Dual Form

The dual becomes:

$$\max_{\alpha} \sum_{i=1}^m \alpha_i - \frac{1}{2}\sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y^{(i)} y^{(j)} K(x^{(i)}, x^{(j)})$$

$$\text{subject to } 0 \leq \alpha_i \leq C, \quad i = 1,\ldots, m$$

$$\sum_{i=1}^m \alpha_i y^{(i)} = 0$$

**Key change:** Now $\alpha_i$ is bounded above by $C$ (instead of $\alpha_i \geq 0$ with no upper bound).

## 8. Key Takeaways

### Why SVMs are Powerful

1. **Large Margin Principle:** Maximizing the margin leads to better generalization
2. **Support Vectors:** Only a few points (support vectors) determine the boundary → robust to most data points
3. **Kernels:** Can learn complex non-linear boundaries efficiently
4. **Soft Margins:** Can handle noisy, non-separable data

### SVM vs Logistic Regression

- **Logistic Regression:** Every point affects the decision boundary; gives probabilities
- **SVM:** Only support vectors matter; focuses on the boundary itself; no probabilities