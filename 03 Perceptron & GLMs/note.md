# **Chapter: The Perceptron & Generalized Linear Models (GLMs)**


## **1. The Perceptron**

### **1.1 Historical Context & Definition**

The Perceptron is one of the earliest supervised learning algorithms (dating back to the 1950s). While less commonly used today in its raw form compared to Logistic Regression or SVMs, it provides the foundational building block for Neural Networks.

It is a binary classification algorithm ($y \in \{0, 1\}$ or $\{-1, 1\}$) that models the decision boundary as a hyperplane.

### **1.2 Hypothesis Function**

Unlike Logistic Regression which outputs a probability via the sigmoid function, the Perceptron uses a hard threshold function (Heaviside step function):

$$h_\theta(x) = g(\theta^T x)$$

Where:

$$g(z) = \begin{cases} 1 & \text{if } z \ge 0 \\ 0 & \text{if } z < 0 \end{cases}$$

### **1.3 Learning Rule (Update Rule)**

The Perceptron learns iteratively. For each training example $(x^{(i)}, y^{(i)})$, the update rule is:

$$\theta_j := \theta_j + \alpha \left( y^{(i)} - h_\theta(x^{(i)}) \right) x_j^{(i)}$$

- If the prediction is correct ($y^{(i)} = h_\theta(x^{(i)})$), the update term is 0; $\theta$ remains unchanged.
    
- If the prediction is wrong:
    
    - **False Negative ($y=1, h=0$):** $\theta := \theta + \alpha x$. (Rotates $\theta$ towards $x$).
        
    - **False Positive ($y=0, h=1$):** $\theta := \theta - \alpha x$. (Rotates $\theta$ away from $x$).
        

**Geometric Intuition:** The weight vector $\theta$ is orthogonal to the decision boundary. When a point is misclassified, adding/subtracting $\alpha x$ to $\theta$ effectively rotates the decision boundary to place $x$ on the correct side.


## **2. The Exponential Family**

To understand why Linear Regression and Logistic Regression work so well and share similar update rules, we look at the broader statistical abstraction: the **Exponential Family** of distributions.

### **2.1 Definition**

A class of distributions is in the Exponential Family if it can be written in the form:

$$p(y; \eta) = b(y) \exp\left( \eta^T T(y) - a(\eta) \right)$$

Where:

- **$\eta$ (eta):** The **Canonical Parameter** (or Natural Parameter).
    
- **$T(y)$:** The **Sufficient Statistic** (often $T(y) = y$).
    
- **$a(\eta)$:** The **Log-Partition Function**. This term ensures the distribution sums/integrates to 1.
    
- **$b(y)$:** The **Base Measure** (usually a constant or scaling factor independent of $\eta$).
    

### **2.2 Properties**

1. **Maximum Likelihood:** Finding the MLE for parameters in an Exponential Family is a convex optimization problem (because $-a(\eta)$ is convex).
    
2. **Moments:** The derivatives of $a(\eta)$ yield the moments of the distribution:
    
    - Expectation: $E[y; \eta] = \frac{\partial}{\partial \eta} a(\eta)$
        
    - Variance: $\text{Var}[y; \eta] = \frac{\partial^2}{\partial \eta^2} a(\eta)$
        

### **2.3 Examples**

- Bernoulli (for Binary Data):
    
    $$p(y; \phi) = \phi^y (1-\phi)^{1-y} = \exp\left( y \ln\left(\frac{\phi}{1-\phi}\right) + \ln(1-\phi) \right)$$
    
    Here: $\eta = \ln(\frac{\phi}{1-\phi})$, $T(y) = y$, $a(\eta) = \ln(1+e^\eta)$ (Softplus function).
    
- **Gaussian (for Continuous Data):** With fixed $\sigma^2$, the Gaussian distribution can be written in this form where $\eta = \mu$.
    

## **3. Generalized Linear Models (GLMs)**

GLMs are a recipe for constructing a learning algorithm for any target variable $y$ that follows a distribution in the Exponential Family.

### **3.1 The Three Assumptions of GLMs**

To derive a GLM for a problem, we make three design choices:

1. **Distribution:** The data $y | x; \theta$ follows an Exponential Family distribution with parameter $\eta$.
    
2. Prediction Goal: Our hypothesis $h_\theta(x)$ outputs the expected value of $y$:
    
    $$h_\theta(x) = E[y | x]$$
    
3. Linear Model: The canonical parameter $\eta$ is linearly related to the input $x$:
    
    $$\eta = \theta^T x$$
    

### **3.2 Deriving Algorithms from GLMs**

- **Ordinary Least Squares (Linear Regression):**
    
    - Assume $y | x \sim \mathcal{N}(\mu, \sigma^2)$.
        
    - In Gaussian, $\mu = \eta$.
        
    - Thus, $h_\theta(x) = E[y|x] = \mu = \eta = \theta^T x$.
        
- **Logistic Regression:**
    
    - Assume $y | x \sim \text{Bernoulli}(\phi)$.
        
    - In Bernoulli, $\eta = \ln(\frac{\phi}{1-\phi}) \implies \phi = \frac{1}{1+e^{-\eta}}$.
        
    - Thus, $h_\theta(x) = E[y|x] = \phi = \frac{1}{1+e^{-\theta^T x}}$ (Sigmoid).
        

### **3.3 The Canonical Link Function**

The function relating the mean $\mu$ to the canonical parameter $\eta$, i.e., $\eta = g(\mu)$, is called the **Link Function**. The inverse, $\mu = g^{-1}(\eta)$, is the **Response Function** (or Transfer Function).


## **4. Softmax Regression (Multinomial Logistic Regression)**

Used for multi-class classification where $y \in \{1, \dots, k\}$.

### **4.1 Multinomial Distribution in Exponential Family**

For $k$ classes, we represent labels as one-hot vectors. The probability is:

$$\phi_1^{1\{y=1\}} \phi_2^{1\{y=2\}} \dots \phi_k^{1\{y=k\}}$$

This can be rewritten in Exponential Family form, leading to the Softmax function as the response function.

### **4.2 Hypothesis & Cost**

- **Hypothesis:** $P(y=i | x; \theta) = \frac{e^{\theta_i^T x}}{\sum_{j=1}^k e^{\theta_j^T x}}$
    
- Cross-Entropy Loss:
    
    $$J(\theta) = - \sum_{i=1}^m \sum_{j=1}^k 1\{y^{(i)} = j\} \log \left( \frac{e^{\theta_j^T x^{(i)}}}{\sum_{l=1}^k e^{\theta_l^T x^{(i)}}} \right)$$
    
    Minimizing this loss (via Gradient Descent) maximizes the likelihood of the training data.