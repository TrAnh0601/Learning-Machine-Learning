# Optimizers & Dropout — Theory and Implementation

> Supplement to `neural_network.py` | CS229 Extension Series

---

## Part 1: Optimizers

### 1.1 The Problem with Inline Updates

In the original design, weight updates were embedded directly inside `Dense.backward()`:

```python
# old — tightly coupled, optimizer is not swappable
def backward(self, output_gradient, learning_rate):
    ...
    self.weights -= learning_rate * weights_gradient
```

Every optimizer beyond vanilla SGD needs to maintain **per-parameter state** — Momentum needs a velocity $v_t$, Adam needs first and second moment estimates $m_t$ and $v_t$ plus a step counter $t$. Storing that state inside the layer pollutes it with optimizer logic, making it impossible to swap optimizers without touching layer code.

The refactored design splits responsibilities cleanly across three components. `Dense.backward()` only computes gradients and stores them in `self._dW` and `self._db`. `Dense.get_params_and_grads()` exposes an iterator of `(param_id, param, grad)` tuples. `NeuralNetwork.train()` calls `optimizer.update()` for each parameter and owns all optimizer state in `self._opt_state`, a dictionary keyed by `(id(layer), 'W')` to avoid collisions between layers of the same shape.

---

### 1.2 SGD (Stochastic Gradient Descent)

**Update rule:**

$$\theta_t = \theta_{t-1} - \alpha \cdot \nabla_\theta \mathcal{L}$$

No state. Each step moves directly along the negative gradient direction.

```python
class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, param, grad, state):
        return param - self.lr * grad, state  # state unchanged
```

**Limitation of SGD:** The gradient at each mini-batch is a noisy estimate of the true gradient. Updates oscillate and converge slowly in ravines — regions where curvature differs sharply across dimensions. This is the core motivation for Momentum and Adam.

---

### 1.3 Momentum

**Intuition:** Instead of following the current gradient directly, accumulate a velocity $v_t$ as an exponentially weighted moving average (EWMA) of past gradients. The optimizer accelerates along directions where gradients are consistent and dampens oscillations along noisy directions — analogous to a ball rolling down a slope that builds speed over time.

**Update rule:**

$$v_t = \beta v_{t-1} + (1 - \beta) \nabla_\theta \mathcal{L}$$

$$\theta_t = \theta_{t-1} - \alpha v_t$$

**Convention note:** Some texts use $v_t = \beta v_{t-1} + \nabla \mathcal{L}$ without the $(1-\beta)$ factor. Both are mathematically equivalent — the scale of $v_t$ differs by a constant that is absorbed into $\alpha$. The $(1-\beta)$ form is preferred here because it keeps the magnitude of $v_t$ consistent with the gradient scale regardless of $\beta$.

```python
class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, beta=0.9):
        self.lr = learning_rate
        self.beta = beta

    def update(self, param, grad, state):
        v = state.get('v', np.zeros_like(param))
        v = self.beta * v + (1 - self.beta) * grad
        return param - self.lr * v, {'v': v}
```

**State:** `{'v': v}` — one array of the same shape as the parameter.

$\beta = 0.9$ means $v_t$ is an EWMA with an effective window of $\approx \frac{1}{1-\beta} = 10$ recent steps.

---

### 1.4 Adam (Adaptive Moment Estimation)

**Idea:** Combine Momentum (1st moment) with a per-parameter adaptive learning rate based on the 2nd moment (uncentered variance of the gradient). Parameters with large and consistent gradients get a smaller effective step; parameters with small or noisy gradients get a larger one. This makes Adam robust to heterogeneous gradient scales across parameters — a common situation in deep networks.

**Update rule** (Kingma & Ba, 2015):

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla_\theta \mathcal{L} \quad \text{(1st moment — mean)}$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla_\theta \mathcal{L})^2 \quad \text{(2nd moment — uncentered variance)}$$

**Bias correction:** At small $t$, both $m_t$ and $v_t$ are biased toward zero because they are initialized at zero. The correction divides by the fraction of mass that has accumulated so far:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$$

```python
class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def update(self, param, grad, state):
        m = state.get('m', np.zeros_like(param))
        v = state.get('v', np.zeros_like(param))
        t = state.get('t', 0) + 1

        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

        m_hat = m / (1 - self.beta1 ** t)
        v_hat = v / (1 - self.beta2 ** t)

        param_new = param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return param_new, {'m': m, 'v': v, 't': t}
```

**State:** `{'m': m, 'v': v, 't': t}` — two arrays plus a scalar step counter.

**Role of $\varepsilon$:** Prevents division by zero when $\hat{v}_t \approx 0$, which can happen in early steps before the second moment has accumulated. $\varepsilon = 10^{-8}$ is the default from the paper.

**Default hyperparameters:** $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\alpha = 0.001$ — taken directly from the paper. These work well across most problems without tuning.

---

### 1.5 Comparison

| Optimizer | State per param | Adaptive lr | Works well for |
|---|---|---|---|
| SGD | None | No | Convex problems; carefully tuned lr |
| Momentum | $v$ | No | Non-convex; faster convergence than SGD |
| Adam | $m$, $v$, $t$ | Yes (per-param) | Default choice; sparse gradients; small datasets |

Adam is a sensible default for most cases. SGD + Momentum is preferred in some vision tasks (e.g. ResNet on ImageNet) because it often generalizes better — a known trade-off between convergence speed and generalization gap that remains an open research question.

---

### 1.6 Critical Bug: `_dW` Must Be Divided by `m`

```python
# wrong — gradient is not normalized over the batch
self._dW = np.dot(self.input.T, output_gradient)

# correct
self._dW = np.dot(self.input.T, output_gradient) / m
```

The cost function is defined as the **mean** loss: $J = \frac{1}{m}\sum \mathcal{L}^{(i)}$. The gradient of a mean is the mean of gradients, so `_dW` must be divided by $m$. Without this, the effective learning rate is $\alpha \cdot m$ — with a large batch this causes immediate divergence. The bug was subtle in this codebase because XOR has $m = 4$, where the inflation is small enough to still converge; it only surfaced on larger datasets.

---

## Part 2: Dropout

### 2.1 Theory

Dropout (Srivastava et al., 2014) is a regularization technique where, during each training forward pass, each neuron is independently zeroed out with probability $1 - p$.

**Why it works:** Dropout prevents neurons from co-adapting — a neuron cannot learn to rely on the presence of specific other neurons, because those neurons may be absent in any given step. Each neuron is forced to learn independently useful features. Theoretically, training with Dropout is equivalent to training an ensemble of $2^n$ sub-networks that share weights, where $n$ is the number of neurons subject to dropout. At inference, the full network approximates the ensemble average.

---

### 2.2 Inverted Dropout

At inference, all neurons are active. If only a $p$ fraction were active during training, the expected output at inference is $1/p$ times larger than what the downstream layers saw during training — a distribution shift that degrades performance.

The naive fix is to scale outputs down by $\times p$ at inference time. This works but requires knowing $p$ in the inference path and modifies what should be a clean, stateless forward pass.

**Inverted Dropout** solves this by scaling *up* by $\times 1/p$ during training instead. The expected value of each activation is preserved throughout training, so inference requires no adjustment at all.

$$\text{output} = \frac{x \cdot \text{mask}}{p}, \quad \text{mask}_i \sim \text{Bernoulli}(p)$$

```python
self._mask = (np.random.rand(*x.shape) < self.keep_prob) / self.keep_prob
return x * self._mask
```

The mask is sampled and immediately divided by `keep_prob` in a single line — this is the entire inverted dropout implementation.

---

### 2.3 Train vs. Inference Mode

Dropout is the first layer in this codebase whose behavior differs between training and inference. BatchNorm has the same property (uses batch statistics at train time, running statistics at test time), so the pattern established here generalizes.

```python
def forward(self, x):
    if not self.training:
        return x       # full pass-through at inference
    self._mask = (np.random.rand(*x.shape) < self.keep_prob) / self.keep_prob
    return x * self._mask
```

The `training` flag is toggled globally by `NeuralNetwork` before each forward pass, so individual layers never need to be managed manually.

> **Common bug:** Calling `model.predict()` inside the training loop — for example, to log loss — sets `training=False` before the backward pass. Dropout then returns `x` directly without setting `self._mask`, leaving `self._mask = None`. The subsequent `backward()` call raises `TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'`. The fix is to run the forward pass manually inside the training loop rather than going through `predict()`.

---

### 2.4 Backward Pass

Dropout is an element-wise multiplication by a fixed mask within a single forward pass, so the chain rule is straightforward:

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\partial \mathcal{L}}{\partial \text{out}_i} \cdot \frac{\text{mask}_i}{p}$$

Since the mask already incorporates the $1/p$ scaling from the forward pass, backward simply multiplies by the same stored mask:

```python
def backward(self, output_gradient):
    return output_gradient * self._mask if self._mask is not None else output_gradient
```

The `None` guard handles the edge case where `backward()` is called after an inference-mode `forward()`.

---

### 2.5 Choosing `keep_prob`

| Layer type | Typical `keep_prob` |
|---|---|
| Hidden dense layers | 0.5 |
| Input layer | 0.8 |
| Recurrent layers | 0.5–0.8 |

`keep_prob = 0.5` is the default from the original paper. For small networks like the one in `neural_network.py`, `0.8–0.9` is safer to avoid underfitting.

---

## Part 3: Design Notes

**Why `param[:] = new_param` instead of `param = new_param`**

After the optimizer computes an updated parameter array, it must be written back so that the layer still holds the new values. `param` inside `NeuralNetwork.train()` is a local reference to `layer.weights`. Reassigning `param = new_param` rebinds the local variable only — `layer.weights` is unchanged. Using `param[:] = new_param` copies data in-place into the buffer that `layer.weights` already points to, so the update is visible through the layer's own reference without any extra bookkeeping.

**Why optimizer state is stored in `NeuralNetwork`, not in each layer**

Storing state in the layer would mean the layer knows which optimizer is being used — coupling two concerns that should be independent. A layer should be purely a differentiable function: it computes a forward pass and a backward pass. The optimizer is a separate algorithm that decides how to use those gradients. Keeping state in `NeuralNetwork._opt_state` (keyed by `(id(layer), param_name)`) lets any optimizer be swapped in without touching any layer code.

---

## References

- Kingma & Ba (2015). *Adam: A Method for Stochastic Optimization.* [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
- Srivastava et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting.* JMLR 15(1).
- Ruder (2016). *An Overview of Gradient Descent Optimization Algorithms.* [ruder.io](https://ruder.io/optimizing-gradient-descent)