# **Chapter: Linear Regression and Gradient Descent**

**Course:** CS229 Machine Learning (Stanford University)

**Instructor:** Andrew Ng

---

## **1. Introduction**

This chapter provides a rigorous foundation for **Linear Regression**, a supervised learning algorithm used to predict continuous target values. Beyond the basic mechanics of the algorithm, we explore its statistical, geometric, and optimization properties, establishing it as the "baseline" for more complex models like Neural Networks.

### **1.1 Notation**

- $m$: Number of training examples.
    
- $n$: Number of features.
    
- $x \in \mathbb{R}^{n+1}$: Input feature vector (including intercept $x_0=1$).
    
- $y \in \mathbb{R}$: Target variable.
    
- $\theta \in \mathbb{R}^{n+1}$: Parameter vector (weights).
    

---

## **2. Linear Regression Model**

### **2.1 Hypothesis**

We approximate the relationship between $x$ and $y$ linearly:

$$h_\theta(x) = \sum_{j=0}^n \theta_j x_j = \theta^T x$$

---

## **3. Cost Function & Probabilistic Interpretation**

### **3.1 The Least Squares Cost Function**

To fit parameters $\theta$, we minimize the **Mean Squared Error (MSE)** cost function:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^m \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2$$

#### **Deep Dive 1: Scaling Factors ($\frac{1}{m}, \frac{1}{2}$)**

- **The factor $\frac{1}{2}$:** Solely for mathematical convenience. When taking the derivative, the exponent $2$ cancels with $\frac{1}{2}$.
    
- **The factor $\frac{1}{m}$:** Normalizes the error by the dataset size.
    
    - **Insight:** Scaling the loss function by a positive constant (like $1/m$ or $1/2$) does **not** change the location of the minimizer $\theta^*$. It only scales the magnitude of the gradient ($\nabla J$).
        
    - **Research Perspective:** If we remove $1/m$, the gradient grows linearly with dataset size $m$. This would require tuning the learning rate $\alpha$ inversely proportional to $m$ to prevent divergence. Keeping $1/m$ makes the choice of $\alpha$ more robust across different dataset sizes.
        

### **3.2 Probabilistic Interpretation (MLE)**

Why squared error? If we assume the target $y$ and inputs $x$ are related via:

$$y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}$$

where noise $\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$ (Gaussian distribution), then the likelihood of the data is:

$$L(\theta) = \prod_{i=1}^m p(y^{(i)} | x^{(i)}; \theta) = \prod_{i=1}^m \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2} \right)$$

Maximizing the Log-Likelihood $\ell(\theta)$ is equivalent to minimizing the term in the exponent, which is exactly the Least Squares Cost Function $J(\theta)$.

- **Research Question:** What if the noise is not Gaussian?
    
    - _Answer:_ If noise follows a Laplace distribution, Maximum Likelihood Estimation leads to **L1 loss** (Mean Absolute Error), which is more robust to outliers (Robust Regression).
        

---

## **4. Convexity Analysis**

Linear Regression is a convex optimization problem.

### **4.1 Hessian Matrix & Curvature**

The gradient of $J(\theta)$ is $\nabla_\theta J = \frac{1}{m} X^T(X\theta - y)$.

The Hessian matrix (second derivative) is:

$$\nabla^2 J(\theta) = \frac{1}{m} X^T X$$

- Insight: For any vector $z \in \mathbb{R}^{n+1}$,
    
    $$z^T (X^T X) z = (Xz)^T (Xz) = \|Xz\|^2 \geq 0$$
    
    Thus, the Hessian is Positive Semidefinite (PSD).
    
- **Implication:** Since $\nabla^2 J(\theta) \succeq 0$, the function $J(\theta)$ is convex (bowl-shaped). It has no local minima; any local minimum is a global minimum.
    
- **Research Note:** If $X^T X$ is Positive Definite (PD) (i.e., invertible, full rank), the solution is unique. If it is only PSD (singular), there are infinite solutions along the "flat" directions (null space), but they all achieve the same minimal loss.
    

---

## **5. Gradient Descent (Numerical Method)**

Update Rule:

$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$$

### **5.1 Feature Scaling & Conditioning**

The convergence speed of Gradient Descent depends heavily on the **Condition Number** of the Hessian $X^T X$, which is the ratio of its largest to smallest eigenvalues ($\kappa = \lambda_{\max} / \lambda_{\min}$).

- **Insight:**
    
    - **Unscaled Features:** If features have vastly different scales (e.g., $x_1 \in [0, 1]$, $x_2 \in [0, 1000]$), the cost contours form elongated ellipses. The gradient direction is not orthogonal to the contours, causing GD to "zig-zag" and converge slowly.
        
    - **Scaled Features:** Contours become closer to circles ($\kappa \approx 1$). The gradient points directly towards the minimum.
        
- **Research Note:** Second-order methods like Newton's Method approximate the inverse Hessian, making them invariant to affine scaling, but they are computationally expensive ($O(n^3)$).
    

---

## **6. The Normal Equations (Closed-Form Solution)**

By setting $\nabla_\theta J(\theta) = 0$, we derive the analytical solution:

$$\theta = (X^T X)^{-1} X^T y$$

### **6.1 Geometric Interpretation**

Linear Regression projects the target vector $y$ onto the **Column Space** of $X$ (denoted $\mathcal{C}(X)$).

- **Prediction:** $\hat{y} = X\theta$. Since $\hat{y}$ is a linear combination of columns of $X$, $\hat{y} \in \mathcal{C}(X)$.
    
- Orthogonality: To minimize the distance $\|y - \hat{y}\|$, the residual vector $r = y - \hat{y}$ must be orthogonal to the subspace $\mathcal{C}(X)$.
    
    $$X^T (y - X\theta) = 0 \implies X^T y = X^T X \theta$$
    
- **Projection Matrix:** The "Hat Matrix" $P = X(X^T X)^{-1}X^T$ projects any vector $y$ onto the span of features $X$.
    

### **6.2 Singularity & Regularization**

If $X^T X$ is non-invertible (singular), standard Linear Regression fails. This happens if $n > m$ or features are linearly dependent.

- Ridge Regression (L2 Regularization): We modify the cost to minimize $J(\theta) + \lambda \|\theta\|^2$. The solution becomes:
    
    $$\theta = (X^T X + \lambda I)^{-1} X^T y$$
    
- **Insight:** Adding $\lambda I$ (where $\lambda > 0$) adds $\lambda$ to all eigenvalues of $X^T X$, ensuring the matrix becomes Positive Definite (invertible).
    
- **Bayesian View:** Ridge Regression corresponds to MAP estimation with a **Gaussian Prior** on $\theta$. L1 Regularization (Lasso) corresponds to a **Laplace Prior**, inducing sparsity.
    

---

## **7. Comparison: Gradient Descent vs. Normal Equations**

|**Feature**|**Gradient Descent (Numerical)**|**Normal Equations (Analytical)**|
|---|---|---|
|**Approach**|Iterative Optimization|Closed-form Algebra / Projection|
|**Complexity**|$O(k \cdot n^2)$ (or $O(k \cdot n)$ with sparse data)|$O(n^3)$ (Matrix Inversion)|
|**Data Size**|Efficient for very large $m$ and $n$.|Slow if $n$ is large ($n > 10,000$).|
|**Hyperparameters**|Requires tuning $\alpha$.|No hyperparameters.|
|**Memory**|Low memory footprint.|Requires storing $X^T X$ ( $O(n^2)$ ).|

**Research Perspective:** Why use GD if we have a closed form?

- When $n$ is massive (e.g., deep learning), inverting a matrix is computationally intractable.
    
- GD allows "online learning" (streaming data).
    
- Iterative methods act as implicit regularization (early stopping).
    

---

## **8. Meta-Insights & Conclusion**

Linear Regression is more than just fitting a line; it serves as a pedagogical bridge connecting multiple fields:

1. **Optimization:** It introduces Convexity, Gradient Descent, and Convergence rates.
    
2. **Linear Algebra:** It demonstrates Projections, Matrix Inversion, Eigenvalues, and Rank.
    
3. **Statistics:** It links Least Squares to Maximum Likelihood Estimation (Gaussian assumption).
    

**Broader Context:**

- **Deep Linear Networks** (Neural Networks with linear activation) collapse mathematically into a single Linear Regression layer, proving that non-linearity is essential for deep learning.
