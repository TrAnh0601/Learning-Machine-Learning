# **Chapter: Locally Weighted Regression & Logistic Regression**

**Course:** CS229 Machine Learning (Stanford University)

**Instructor:** Andrew Ng



## **1. Locally Weighted Regression (LWR)**

### **1.1 Motivation: Underfitting vs. Overfitting**

In Linear Regression, choosing the right features is critical.

- **Underfitting (High Bias):** Using a linear model $y = \theta_0 + \theta_1 x$ for data that has a curved structure.
    
- **Overfitting (High Variance):** Using a high-degree polynomial (e.g., $y = \sum_{j=0}^{10} \theta_j x^j$) which passes through every point but fails to generalize.
    

Instead of manually engineering features (like $x^2, \sqrt{x}$), **Locally Weighted Regression (LWR)** allows the model to fit a flexible curve by focusing only on data points nearby the query point $x$.

### **1.2 Cost Function & Weighting**

To make a prediction at a specific query point $x$, LWR minimizes a weighted cost function:

$$\min_\theta \sum_{i=1}^m w^{(i)} \left( y^{(i)} - \theta^T x^{(i)} \right)^2$$

Where the weight $w^{(i)}$ depends on the distance between the training example $x^{(i)}$ and the query point $x$:

$$w^{(i)} = \exp\left( -\frac{(x^{(i)} - x)^2}{2\tau^2} \right)$$

- If $|x^{(i)} - x|$ is small (close): $w^{(i)} \approx 1$.
    
- If $|x^{(i)} - x|$ is large (far): $w^{(i)} \approx 0$.
    
- **$\tau$ (Bandwidth parameter):** Controls how quickly weights fall off. A small $\tau$ makes the model "local" (sensitive to noise), while a large $\tau$ makes it smoother (closer to standard linear regression).
    

### **1.3 Parametric vs. Non-parametric Learning**

- **Parametric (e.g., Linear Regression):** The number of parameters $\theta$ is fixed and independent of the training set size $m$. Once $\theta$ is learned, training data can be discarded.
    
- **Non-parametric (e.g., LWR):** The number of parameters (or the complexity) grows with $m$. We must keep the entire training set in memory to make predictions (since $w^{(i)}$ depends on $x^{(i)}$ and the _current_ query $x$).
    


## **2. Probabilistic Interpretation of Linear Regression**

Why do we use the Least Squares cost function?

Assume the relationship between $x$ and $y$ is:

$$y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}$$

where $\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$ (Gaussian noise).

The probability density of $y^{(i)}$ given $x^{(i)}$ is:

$$p(y^{(i)} | x^{(i)}; \theta) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2} \right)$$

Likelihood Function:

$$L(\theta) = \prod_{i=1}^m p(y^{(i)} | x^{(i)}; \theta)$$

Log-Likelihood:

$$\ell(\theta) = \log L(\theta) = \sum_{i=1}^m \log \left( \frac{1}{\sqrt{2\pi}\sigma} \right) - \frac{1}{2\sigma^2} \sum_{i=1}^m (y^{(i)} - \theta^T x^{(i)})^2$$

Maximizing $\ell(\theta)$ (Maximum Likelihood Estimation - MLE) is equivalent to minimizing the term $\sum (y^{(i)} - \theta^T x^{(i)})^2$, which is exactly the **Least Squares Cost Function**.



## **3. Logistic Regression**

Designed for Classification problems where $y \in \{0, 1\}$.

(Note: Linear Regression is bad for classification because predicted values can be $>1$ or $<0$, and MSE is non-convex for classification).

### **3.1 Hypothesis: The Sigmoid Function**

We want $0 \leq h_\theta(x) \leq 1$. We use the Logistic (Sigmoid) function:

$$h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

where $g(z) = \frac{1}{1+e^{-z}}$.

- **Property:** $g'(z) = g(z)(1 - g(z))$ (Useful for derivation).
    

### **3.2 Probabilistic Assumption**

$$P(y=1 | x; \theta) = h_\theta(x)$$

$$P(y=0 | x; \theta) = 1 - h_\theta(x)$$

Combined form (Bernoulli distribution):

$$p(y | x; \theta) = (h_\theta(x))^y (1 - h_\theta(x))^{1-y}$$

### **3.3 Cost Function (Maximum Likelihood)**

Likelihood: $L(\theta) = \prod_{i=1}^m (h_\theta(x^{(i)}))^{y^{(i)}} (1 - h_\theta(x^{(i)}))^{1-y^{(i)}}$

Log-Likelihood:

$$\ell(\theta) = \sum_{i=1}^m \left[ y^{(i)} \log h_\theta(x^{(i)}) + (1 - y^{(i)}) \log (1 - h_\theta(x^{(i)})) \right]$$

To find $\theta$, we use Gradient Ascent to maximize $\ell(\theta)$.

Update rule:

$$\theta_j := \theta_j + \alpha \frac{\partial \ell(\theta)}{\partial \theta_j}$$

Derivative Derivation:

After simplifying (using $g' = g(1-g)$):

$$\frac{\partial \ell(\theta)}{\partial \theta_j} = \sum_{i=1}^m \left( y^{(i)} - h_\theta(x^{(i)}) \right) x_j^{(i)}$$

Thus, the Stochastic Gradient Ascent rule is:

$$\theta_j := \theta_j + \alpha (y^{(i)} - h_\theta(x^{(i)})) x_j^{(i)}$$

- **Remark:** This looks identical to the Linear Regression update rule, but here $h_\theta(x)$ is the sigmoid function, not linear.
    


## **4. Newton's Method**

An iterative method to find the root of a function $f(\theta) = 0$.

To maximize $\ell(\theta)$, we find the root of its derivative: $f(\theta) = \ell'(\theta) = 0$.

### **4.1 Update Rule**

Newton's update for finding zeros of $\ell'(\theta)$:

$$\theta := \theta - \frac{\ell'(\theta)}{\ell''(\theta)}$$

In multi-dimensional settings (vector $\theta$):

$$\theta := \theta - H^{-1} \nabla_\theta \ell(\theta)$$

Where $H$ is the Hessian Matrix (matrix of second derivatives):

$$H_{ij} = \frac{\partial^2 \ell(\theta)}{\partial \theta_i \partial \theta_j}$$

### **4.2 Properties**

- **Quadratic Convergence:** Extremely fast convergence (the number of significant digits doubles each iteration). Usually converges in 5-15 iterations.
    
- **Computational Cost:** Requires inverting the Hessian $H$ (size $n \times n$), which costs $O(n^3)$. Suitable only for small to medium feature sets ($n < 1000$).