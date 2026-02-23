# AdaBoost: Theoretical & Mathematical Notes

> **Scope:** AdaBoost.M1 (binary classification). Notation follows Freund & Schapire (1997) and Friedman, Hastie & Tibshirani (2000).


## 1. Problem Setup

Given dataset $\{(x_i, y_i)\}_{i=1}^{N}$, $y_i \in \{-1, +1\}$, the goal is to construct a strong classifier:

$$H(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t h_t(x)\right) = \text{sign}(F(x))$$

where each $h_t : \mathcal{X} \to \{-1, +1\}$ is a **weak learner** (e.g., decision stump), and $\alpha_t \in \mathbb{R}$ is its contribution weight.

**Weak learning assumption:** Each $h_t$ achieves weighted error $\epsilon_t < 0.5$ — i.e., strictly better than random on the current distribution. This is the only requirement AdaBoost places on the base learner.


## 2. Algorithm

### Initialization
$$D_1(i) = \frac{1}{N} \quad \forall i$$

### Per Iteration $t = 1, \ldots, T$

**Step 1 — Resample (original formulation):**

$$\mathcal{S}_t = \text{Resample}(X, D_t, N) \quad \text{with replacement}$$

The resampled set encodes $D_t$ implicitly via frequency. Training $h_t$ on $\mathcal{S}_t$ with uniform weights is equivalent *in expectation* to training with explicit sample weights — but resampling introduces Monte Carlo noise that can act as mild regularization.

> **Weighted-loss variant (e.g., sklearn):** Skip resampling; pass $D_t$ as `sample_weight` directly into the base learner. Exact rather than approximate, preferred when the learner supports weighted training.

**Step 2 — Train weak learner:**

$$h_t = \arg\min_{h \in \mathcal{H}} \sum_{i=1}^{N} D_t(i) \cdot \mathbf{1}[h(x_i) \neq y_i]$$

For decision stumps: exhaustive search over all (feature, threshold, polarity) triples. Evaluation of $\epsilon_t$ must use the **original** $D_t$, not the resampled distribution.

**Step 3 — Compute learner weight:**

$$\epsilon_t = \sum_{i=1}^{N} D_t(i) \cdot \mathbf{1}[h_t(x_i) \neq y_i]$$

$$\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$$

Properties:
- $\epsilon_t < 0.5 \Rightarrow \alpha_t > 0$: correct learners contribute positively
- $\epsilon_t \to 0 \Rightarrow \alpha_t \to \infty$: perfect learners dominate
- $\epsilon_t = 0.5 \Rightarrow \alpha_t = 0$: random learner is ignored

**Step 4 — Update distribution:**

$$D_{t+1}(i) = \frac{D_t(i) \cdot \exp(-\alpha_t \, y_i \, h_t(x_i))}{Z_t}$$

where $Z_t = \sum_i D_t(i) \exp(-\alpha_t \, y_i \, h_t(x_i))$ is the normalization constant.

Intuition:
- $y_i h_t(x_i) = +1$ (correct) $\Rightarrow$ weight decreases by $e^{-\alpha_t}$
- $y_i h_t(x_i) = -1$ (wrong) $\Rightarrow$ weight increases by $e^{+\alpha_t}$

**Step 5 — Final prediction:**

$$H(x) = \text{sign}\!\left(\sum_{t=1}^{T} \alpha_t h_t(x)\right)$$


## 3. Loss Function: Exponential Loss

AdaBoost implicitly minimizes **exponential loss** via coordinate descent in function space (Friedman et al., 2000):

$$\mathcal{L}(F) = \sum_{i=1}^{N} \exp(-y_i F(x_i)), \quad F(x) = \sum_t \alpha_t h_t(x)$$

At each iteration, AdaBoost finds $(\alpha_t, h_t)$ that greedily minimizes $\mathcal{L}(F_{t-1} + \alpha h)$:

$$(\alpha_t, h_t) = \arg\min_{\alpha, h} \sum_i \exp\!\left(-y_i (F_{t-1}(x_i) + \alpha h(x_i))\right)$$

Expanding:

$$= \sum_i \underbrace{\exp(-y_i F_{t-1}(x_i))}_{w_i^{(t)}} \exp(-\alpha \, y_i \, h(x_i))$$

Separating correct/incorrect predictions and optimizing over $\alpha$ analytically recovers exactly $\alpha_t$ and the weight update of the algorithm. This confirms that **sample weights $D_t$ are (normalized) pointwise loss values** $w_i^{(t)} = \exp(-y_i F_{t-1}(x_i))$.

### Connection to Gradient Boosting

AdaBoost is a special case of **Gradient Boosting** (Friedman, 2001) with exponential loss. The pseudo-residuals under exponential loss are:

$$r_i = -\frac{\partial \mathcal{L}}{\partial F(x_i)} = y_i \exp(-y_i F(x_i)) = y_i \cdot w_i^{(t)}$$

Fitting $h_t$ to minimize weighted error on $D_t$ is equivalent to fitting $h_t$ to approximate these pseudo-residuals. This unification implies:

> **AdaBoost = Gradient Boosting with exponential loss + closed-form step size.**

Gradient Boosting generalizes this by allowing any differentiable loss and using a fixed learning rate $\eta$ instead of the analytically derived $\alpha_t$.


## 4. Training Error Bound

**Theorem (Freund & Schapire, 1997):**

$$\text{TrainError}(H) \leq \prod_{t=1}^{T} Z_t$$

Since $Z_t = 2\sqrt{\epsilon_t(1-\epsilon_t)}$, and letting $\gamma_t = \frac{1}{2} - \epsilon_t$ (edge over random):

$$\prod_{t=1}^{T} Z_t = \prod_{t=1}^{T} \sqrt{1 - 4\gamma_t^2} \leq \exp\!\left(-2\sum_{t=1}^{T} \gamma_t^2\right)$$

If each weak learner maintains a constant edge $\gamma_t \geq \gamma > 0$:

$$\text{TrainError}(H) \leq \exp(-2\gamma^2 T)$$

Training error decreases **exponentially** in $T$ — AdaBoost can drive training error to zero given sufficient weak learners.


## 5. Generalization: Margin Theory

AdaBoost often avoids overfitting even after training error reaches zero. Explanation via **margin theory** (Schapire et al., 1998):

**Margin** of example $(x_i, y_i)$:

$$\rho_i = \frac{y_i \sum_t \alpha_t h_t(x_i)}{\sum_t \alpha_t} \in [-1, +1]$$

**Margin bound:**

$$\mathbb{P}_{\text{test}}[H(x) \neq y] \leq \mathbb{P}_{\text{train}}[\rho_i \leq \theta] + O\!\left(\sqrt{\frac{\log T}{N\theta^2}}\right)$$

After training error = 0, AdaBoost continues to **increase margins** — effectively performing max-margin optimization, analogous to SVMs. This explains why more iterations can *improve* generalization even with zero training error on clean data.

**Breakdown with label noise:** Noisy samples receive exponentially growing weights, driving $D_t$ to concentrate on noise. The margin bound no longer holds — overfitting follows. This is the fundamental fragility of exponential loss.


## 6. Bias–Variance Perspective

| Component | Effect |
|---|---|
| Few iterations ($T$ small) | High bias, underfitting |
| Many iterations ($T$ large, clean data) | Increasing margins, often no overfit |
| Many iterations ($T$ large, noisy data) | Variance explodes, exponential loss concentrates on noise |
| Shallow trees (stumps) | Strong inductive bias: axis-aligned boundaries, no feature interactions |
| Deeper trees | Capture interactions, but less "weak" — boosting less effective |

AdaBoost with stumps has high bias in each learner but reduces bias exponentially through aggregation — the classic **bias reduction** mechanism of boosting, as opposed to Random Forest's **variance reduction**.


## 7. Regularization

AdaBoost has **no explicit regularization**. The primary implicit regulators are:

- **$T$ (number of estimators):** Main knob. Early stopping is the only reliable defense against noisy-label overfitting.
- **Tree depth:** Stumps are the strongest regularizer via inductive bias. Increasing depth increases capacity.
- **Resampling noise:** Sampling variance from resampling acts as very mild regularization (not reliable).

Compare to XGBoost which adds $\Omega(f_t) = \gamma T + \frac{1}{2}\lambda\|w\|^2$ explicitly into the objective.


## 8. Key Limitations

| Limitation | Root Cause |
|---|---|
| Outlier / label noise sensitivity | Exponential loss: unbounded penalty on persistent errors |
| Sequential training | Each $h_t$ depends on $D_t$; not parallelizable (unlike Random Forest) |
| No calibrated probabilities | $F(x)$ is a score, not a probability; requires Platt scaling |
| Fixed loss function | Cannot plug in arbitrary losses (unlike GBM) |