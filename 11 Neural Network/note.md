# Chapter 11: Introduction to Neural Networks


## 1. Foundations of Deep Learning

Deep Learning is a subset of machine learning that has revolutionized computer vision, NLP, and speech recognition. Success is driven by three factors:

- **Computational Power:** Parallelized large-scale matrix operations on GPUs.
- **Data Availability:** Exponential growth of digitized data enables flexible models to learn complex features.
- **Algorithmic Innovation:** Techniques to train deeper architectures effectively (better initialization, normalization, activations).


## 2. Logistic Regression as a Single Neuron

The simplest neural network: a single neuron performing binary classification.

### 2.1 Model Components

**Linear step:**
$$z = w^T x + b, \quad w \in \mathbb{R}^n,\ b \in \mathbb{R}$$

**Activation step (Sigmoid):**
$$a = \sigma(z) = \frac{1}{1 + e^{-z}}$$

**Gradient of sigmoid** (critical for backprop):
$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

> **Vanishing gradient:** At saturation regions ($z \gg 0$ or $z \ll 0$), $\sigma'(z) \approx 0$, causing gradients to vanish during backpropagation. This is a fundamental reason sigmoid is avoided in hidden layers of deep networks.

### 2.2 Loss Function — MLE Derivation

Binary Cross-Entropy (BCE) is not chosen arbitrarily — it follows directly from **Maximum Likelihood Estimation** under a Bernoulli distribution.

Assume $\hat{y} = P(y=1 \mid x)$. The likelihood for one example:
$$P(y \mid x) = \hat{y}^y (1 - \hat{y})^{1-y}$$

Taking the negative log-likelihood:
$$\mathcal{L}(\hat{y}, y) = -\left[ y \log \hat{y} + (1 - y) \log(1 - \hat{y}) \right]$$

> **Why not MSE for classification?** MSE assumes Gaussian noise (from MLE under Gaussian likelihood). For binary outputs, Bernoulli is the correct distributional assumption → BCE. MSE also leads to non-convex loss surfaces with sigmoid activations, causing poor gradient flow early in training.

### 2.3 Case Study: Image Classification

For a $64 \times 64$ RGB image:
$$n = 64 \times 64 \times 3 = 12{,}288$$

The input vector $x \in \mathbb{R}^{12288}$ is formed by flattening the 3D tensor. Weight vector $w \in \mathbb{R}^{12288}$ (or equivalently, $W$ of shape $1 \times 12288$ in matrix form).


## 3. Neural Network Architectures

### 3.1 Multi-Class & Multi-Label Classification

| Setting | Activation | When to use |
|---|---|---|
| **Multi-label** | Sigmoid per output neuron | Classes are independent (multiple can be true) |
| **Multi-class** | Softmax over all outputs | Classes are mutually exclusive |

**Softmax:**
$$a_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$$

**Numerical stability — Log-Sum-Exp trick:**

Naive Softmax overflows for large $z_i$. Stable implementation:
$$a_i = \frac{e^{z_i - \max_j z_j}}{\sum_{j=1}^{C} e^{z_j - \max_j z_j}}$$

This is mathematically equivalent (constant cancels) but avoids `inf` in `exp`.

**Softmax + Cross-Entropy gradient** (clean and critical result):

With loss $\mathcal{L} = -\sum_k y_k \log a_k$:
$$\frac{\partial \mathcal{L}}{\partial z_i} = a_i - y_i$$

> This is not obvious — it requires combining the Softmax Jacobian with the cross-entropy gradient. The result is elegant: gradient is simply the prediction error. Worth deriving at least once.

### 3.2 Notation for Deep Layers

| Symbol | Meaning |
|---|---|
| $W^{[l]},\ b^{[l]}$ | Parameters of layer $l$ |
| $a^{[l]}_i$ | Output of $i$-th neuron in layer $l$ |
| $x^{(i)}$ | $i$-th training example |
| $n^{[l]}$ | Number of neurons in layer $l$ |


## 4. Mathematical Formulation: Forward Propagation

For layer $l$, computed recursively:

$$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$
$$a^{[l]} = g^{[l]}(z^{[l]})$$

where $a^{[0]} = x$ and $g^{[l]}$ is the activation function.

### 4.1 Matrix Shapes

If layer $l$ has $n^{[l]}$ neurons and layer $l-1$ has $n^{[l-1]}$ neurons:

| Quantity | Shape |
|---|---|
| $W^{[l]}$ | $(n^{[l]} \times n^{[l-1]})$ |
| $b^{[l]}$ | $(n^{[l]} \times 1)$ |
| $z^{[l]},\ a^{[l]}$ | $(n^{[l]} \times 1)$ |

**Intuition:** Each **row** of $W^{[l]}$ is one neuron — a linear classifier operating on the previous layer's activations. This framing generalizes naturally to Conv layers (where neurons share weights spatially).


## 5. Activation Functions

Non-linearity is mandatory — without it, a deep network collapses to a single affine transformation regardless of depth.

### 5.1 Comparison

| Activation | Formula | Range | Gradient | Notes |
|---|---|---|---|---|
| **Sigmoid** | $\frac{1}{1+e^{-z}}$ | $(0, 1)$ | $\sigma(1-\sigma) \in (0, 0.25]$ | Vanishing grad; not zero-centered |
| **Tanh** | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $(-1, 1)$ | $1 - \tanh^2(z) \in (0, 1]$ | Zero-centered; still saturates |
| **ReLU** | $\max(0, z)$ | $[0, \infty)$ | $\mathbf{1}[z > 0]$ | No saturation for $z>0$; dying ReLU problem |
| **Leaky ReLU** | $\max(\alpha z, z)$, $\alpha \ll 1$ | $\mathbb{R}$ | $1$ or $\alpha$ | Fixes dying ReLU |
| **GELU** | $z \cdot \Phi(z)$ | $\mathbb{R}$ | smooth approx. | Default in Transformers (BERT, GPT) |

### 5.2 Dying ReLU Problem

When $z < 0$ for a neuron across all training examples, its gradient is permanently 0 → neuron never updates. Caused by large negative bias or large learning rate. **Fix:** Leaky ReLU, He initialization.

### 5.3 Why Depth Requires ReLU (Not Sigmoid/Tanh)

With $L$ layers using sigmoid: gradient magnitude $\propto (0.25)^L$ → exponential decay. ReLU preserves gradient magnitude (= 1) for active neurons, enabling training of very deep networks.


## 6. Weight Initialization

### 6.1 Symmetry Problem

If all weights initialized to the same value (e.g., 0), all neurons in a layer compute identical gradients → they learn identical features forever. **Solution:** random initialization.

### 6.2 Variance Scaling — Xavier & He Initialization

Naive random init (e.g., $\mathcal{N}(0,1)$) causes variance of activations to explode or vanish with depth. Proper initialization keeps variance stable across layers.

**Derivation intuition:** For a layer with $n^{[l-1]}$ inputs, if weights $w \sim \mathcal{N}(0, \sigma^2)$ and inputs have unit variance:
$$\text{Var}(z) = n^{[l-1]} \cdot \sigma^2 \cdot \text{Var}(a^{[l-1]})$$

To keep $\text{Var}(z) = \text{Var}(a^{[l-1]})$, set $\sigma^2 = \frac{1}{n^{[l-1]}}$.

| Init | Formula | Best for |
|---|---|---|
| **Xavier / Glorot** | $\sigma^2 = \frac{2}{n^{[l-1]} + n^{[l]}}$ | Sigmoid, Tanh |
| **He** | $\sigma^2 = \frac{2}{n^{[l-1]}}$ | ReLU, Leaky ReLU |

> He init accounts for the fact that ReLU zeros out ~50% of activations, effectively halving the variance contribution.


## 7. Vectorization and Batching

### 7.1 Batch Forward Pass

Stack $m$ training examples column-wise: $X \in \mathbb{R}^{n \times m}$.

$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$$
$$A^{[l]} = g^{[l]}(Z^{[l]})$$

Shapes:

| Quantity | Shape |
|---|---|
| $X = A^{[0]}$ | $(n^{[0]} \times m)$ |
| $Z^{[l]},\ A^{[l]}$ | $(n^{[l]} \times m)$ |
| $W^{[l]}$ | $(n^{[l]} \times n^{[l-1]})$ |
| $b^{[l]}$ | $(n^{[l]} \times 1)$ → broadcast to $(n^{[l]} \times m)$ |

**Why column-stacking?** Convention aligns with BLAS/LAPACK column-major operations and is consistent with PyTorch's convention (`batch_size` as first dim after transpose). In practice, frameworks use $(m \times n)$ (row-major), so be careful when translating math to code.

### 7.2 Mini-batch vs SGD vs Full Batch

| Method | Batch size | Gradient estimate | Notes |
|---|---|---|---|
| **SGD** | 1 | High variance | Noisy updates; can escape local minima |
| **Mini-batch GD** | 32–512 | Moderate variance | Best hardware utilization; standard practice |
| **Full-batch GD** | $m$ | Exact | Expensive; converges smoothly but can overfit |

Mini-batch introduces noise that acts as implicit regularization — a key insight for understanding generalization in deep learning.


## 8. Backpropagation

The mechanism for computing $\frac{\partial \mathcal{L}}{\partial W^{[l]}}$ efficiently via the **chain rule** over the computational graph.

### 8.1 Computational Graph View

Each forward pass operation is a node. Backprop traverses this graph in reverse, accumulating gradients. This is what PyTorch's `autograd` implements — understanding this is essential for debugging gradient flow.

### 8.2 General Backprop Equations (Single Example)

Define the **error signal** at layer $l$:
$$\delta^{[l]} = \frac{\partial \mathcal{L}}{\partial z^{[l]}}$$

**Output layer** (with Cross-Entropy + Softmax):
$$\delta^{[L]} = a^{[L]} - y$$

**Hidden layer** (recursive):
$$\delta^{[l]} = \left(W^{[l+1]T} \delta^{[l+1]}\right) \odot g'^{[l]}(z^{[l]})$$

where $\odot$ is element-wise multiplication.

**Parameter gradients:**
$$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \delta^{[l]} a^{[l-1]T}$$
$$\frac{\partial \mathcal{L}}{\partial b^{[l]}} = \delta^{[l]}$$

**Vectorized over batch** ($m$ examples):
$$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \frac{1}{m} \Delta^{[l]} A^{[l-1]T}$$;
$$\frac{\partial \mathcal{L}}{\partial b^{[l]}} = \frac{1}{m} \sum_{i=1}^{m} \delta^{[l]\(i)}$$

### 8.3 Vanishing & Exploding Gradients

From the recursive formula $\delta^{[l]} = W^{[l+1]T} \delta^{[l+1]} \odot g'$:

- **Vanishing:** If $\|W\|$ is small or $g'$ saturates (sigmoid) → $\|\delta^{[l]}\| \to 0$ exponentially with depth → early layers learn nothing.
- **Exploding:** If $\|W\|$ is large → $\|\delta^{[l]}\|$ grows exponentially → unstable training, `NaN` loss.

**Mitigations:**

| Problem | Solution |
|---|---|
| Vanishing | ReLU activations, He init, residual connections (ResNet), BatchNorm |
| Exploding | Gradient clipping, careful init, weight decay |


## 9. Optimization: The Training Loop

**Cost function** (average loss over $m$ examples):
$$J(W, b) = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(\hat{y}^{(i)}, y^{(i)})$$

**Training loop:**

1. **Initialize** parameters (Xavier or He init).
2. **Forward pass** — compute $\hat{y}$ and $J$.
3. **Backward pass** — compute all gradients via backprop.
4. **Update** (Gradient Descent):
$$W^{[l]} := W^{[l]} - \alpha \frac{\partial J}{\partial W^{[l]}}$$

where $\alpha$ is the learning rate.


## 10. Universal Approximation Theorem

**Statement:** A feedforward network with a single hidden layer of sufficient width and a non-linear activation can approximate any continuous function on a compact subset of $\mathbb{R}^n$ to arbitrary precision.

**Implications and limitations for research:**

- Guarantees expressiveness *in theory* — does not say anything about **learnability** (optimization) or **generalization** (sample efficiency).
- Width required may be exponential in input dimension — not practical.
- **Depth is more efficient than width:** Deep networks can represent functions that require exponentially more neurons if implemented shallow (depth-separation theorems).
- This is why deep architectures dominate in practice — not just empirically, but with theoretical backing.


## 11. Meta-Insights & Research Extensions

### 11.1 Representation Learning

Neural networks act as a hierarchy of feature extractors:
- **Early layers:** simple primitives (edges, frequencies).
- **Middle layers:** textures, parts.
- **Deep layers:** semantic concepts (faces, objects).

This **representational hierarchy** is what makes transfer learning possible — features learned on ImageNet transfer to medical imaging because low/mid-level features are domain-agnostic.

### 11.2 End-to-End Learning

Raw input → prediction, without manual feature engineering. The network discovers the optimal representation for the task. Contrast with classical ML pipelines: hand-crafted features → classifier.

### 11.3 Inductive Bias: MLP vs CNN vs Transformer

A core concept in research — every architecture encodes assumptions about the structure of data.

| Architecture | Inductive Bias | Data efficiency |
|---|---|---|
| **MLP** | None (fully connected) | Low — needs much data to discover structure |
| **CNN** | Locality + translation equivariance | High for spatial data (images) |
| **Transformer** | Pairwise attention (relational) | Medium — learns global structure, scales with data |
| **GNN** | Graph topology | High for graph-structured data |

> MLP-Mixer (Tolstikhin et al., 2021) is an interesting data point: a pure MLP architecture competitive with ViT on ImageNet, showing that with enough data, learned structure can substitute for inductive bias.

### 11.4 Non-Linearity — Formal Necessity

Without activation functions, for any depth $L$:
$$a^{[L]} = W^{[L]} W^{[L-1]} \cdots W^{[1]} x = W_{\text{eff}} x$$

The composition collapses to a single linear map. Adding even one non-linear layer makes the function class qualitatively richer (non-convex decision boundaries).
