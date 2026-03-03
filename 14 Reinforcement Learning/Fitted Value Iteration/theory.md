# Chapter: Continuous State MDPs & Model-Based Reinforcement Learning


## 1. Problem Setup: Continuous State Spaces

In prior lectures, MDPs had **finite discrete** state spaces (e.g., an 11-state grid). Real-world control problems operate in **continuous** $\mathcal{S} \subseteq \mathbb{R}^n$, making tabular methods infeasible.

### 1.1 Motivating Examples

| System | State Space $s$ | Dim $n$ |
|---|---|---|
| **Self-driving car** | $(x, y, \theta, \dot{x}, \dot{y}, \dot{\theta})$ | 6 |
| **Inverted pendulum** | $(x, \theta, \dot{x}, \dot{\theta})$ | 4 |
| **Autonomous helicopter** | $(x,y,z, \phi, \theta, \psi, \dot{x}, \dot{y}, \dot{z}, \dot{\phi}, \dot{\theta}, \dot{\psi})$ | 12 |


## 2. Discretization & The Curse of Dimensionality

The simplest approach: discretize each dimension into $k$ values → apply standard Value/Policy Iteration.

### Why It Fails

Discretizing $n$ dimensions into $k$ bins yields $|\mathcal{S}| = k^n$ states.

$$k = 10,\; n = 12 \;\Longrightarrow\; |\mathcal{S}| = 10^{12} \quad \text{(computationally intractable)}$$

**Rule of thumb:** Discretization is only practical for $n \leq 3$.

Beyond the storage issue, discretization introduces **representation error** — two distinct continuous states mapped to the same bin are treated identically, causing information loss that degrades policy quality.


## 3. Model-Based Reinforcement Learning

Instead of discretizing, **Model-Based RL (MBRL)** learns a **dynamics model** (simulator):

$$s_{t+1} = f(s_t, a_t)$$

Then uses this model to perform planning (compute a good policy) without further real-world interaction.

### 3.1 Two Ways to Obtain a Model

**1. Physics-based simulation**  
Encode Newton's equations or use an existing physics engine. Example for a simple 1D system:

$$s' = s + \Delta t \cdot \dot{s}$$

**2. System Identification (learn from data)**  
- Execute a policy on the real robot for $T$ timesteps, repeat $m$ trials
- Collect trajectory data: $(s_0, a_0, s_1),\, (s_1, a_1, s_2), \ldots$
- Treat as a supervised learning problem: **input** $(s_t, a_t)$, **target** $s_{t+1}$

### 3.2 Linear Dynamics Model

The simplest parametric dynamics:

$$s_{t+1} \approx As_t + Ba_t$$

where $A \in \mathbb{R}^{n \times n}$, $B \in \mathbb{R}^{n \times d}$. Fit by least squares:

$$\min_{A,\, B} \sum_{i=1}^{m} \sum_{t=0}^{T-1} \left\| s_{t+1}^{(i)} - \left(A s_t^{(i)} + B a_t^{(i)}\right) \right\|^2$$

> **Limitation:** Most real-world dynamics are **nonlinear**. Linear models are good as a baseline but fail for systems with complex contact dynamics or aerodynamics. Modern approaches use neural network dynamics models — see [§6](#6-limitations--connections-to-modern-mbrl).

### 3.3 Stochastic vs. Deterministic Models

A deterministic model $s_{t+1} = As_t + Ba_t$ produces policies that are **brittle** — they overfit to the model's perfect predictions and collapse under real-world noise.

**Fix:** Add a stochastic noise term:

$$s_{t+1} = As_t + Ba_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \Sigma)$$

Training on a stochastic simulator forces the value function $V(s)$ to be **robust** — it cannot rely on a single "perfect" trajectory, so it learns to hedge against model error.

> **Key design principle (Andrew Ng):** *"Train on a stochastic simulator; deploy with a deterministic policy."*  
> This is an early instance of what is now formalized as **domain randomization** in sim-to-real transfer.


## 4. Fitted Value Iteration (FVI)

FVI extends Value Iteration to continuous spaces by **approximating** $V(s)$ with a parametric function:

$$V(s) \approx \theta^\top \phi(s)$$

where $\phi(s) \in \mathbb{R}^k$ are hand-crafted features (e.g., $\phi(s) = [x,\, \dot{x},\, \theta,\, \dot{\theta},\, x\theta,\, \ldots]^\top$) and $\theta \in \mathbb{R}^k$ are learned weights.

### 4.1 Algorithm

**Input:** A set of $m$ sampled states $\{s^{(1)}, \ldots, s^{(m)}\} \subset \mathbb{R}^n$, a simulator $f$, reward $R$, discount $\gamma$.


**Initialization:** $\theta \leftarrow \mathbf{0}$

**Repeat until convergence:**

For each sampled state $s^{(i)}$, for each action $a$:

1. **Sample** $k$ next states via stochastic simulator:
$$s'_1, \ldots, s'_k \sim P_{s^{(i)} a}$$

2. **Estimate** the action-value (Monte Carlo expectation):
$$Q(s^{(i)}, a) = R(s^{(i)}) + \gamma \cdot \frac{1}{k} \sum_{j=1}^{k} \theta^\top \phi(s'_j)$$

3. **Compute Bellman target:**
$$y^{(i)} = \max_a\, Q(s^{(i)}, a)$$

**Regression step** — fit $\theta$ to targets $\{(s^{(i)}, y^{(i)})\}$:

$$\theta \leftarrow \arg\min_\theta \frac{1}{2} \sum_{i=1}^{m} \left(\theta^\top \phi(s^{(i)}) - y^{(i)}\right)^2$$


### 4.2 Monte Carlo Expectation: Choosing $k$

| Model type | Recommended $k$ | Rationale |
|---|---|---|
| Deterministic | $k = 1$ | $P_{sa}$ is a point mass; one simulation suffices |
| Stochastic (low noise) | $k = 3$–$5$ | Low variance, small overhead |
| Stochastic (high noise) | $k \geq 10$ | High bias if $k$ too small |

> **Bias-variance tradeoff:** Small $k$ → high variance estimates of $\mathbb{E}[V(s')]$ → noisy regression targets → slow / unstable convergence. Large $k$ → computational cost scales linearly.

### 4.3 Role of Feature Engineering $\phi(s)$

The quality of FVI is **directly bounded** by the expressiveness of $\phi(s)$. With linear function approximation, $V^\pi$ must be approximately linear in $\phi(s)$ for FVI to produce a near-optimal policy. This is a strong assumption.

Common choices:
- **Polynomial features:** $1, x, x^2, x\theta, \ldots$
- **Radial Basis Functions (RBF):** $\phi_j(s) = \exp\!\left(-\|s - \mu_j\|^2 / 2\sigma^2\right)$
- **Neural network:** Replace $\theta^\top \phi(s)$ with $f_\theta(s)$ → this is exactly **DQN**

### 4.4 Convergence Warning

Unlike tabular Value Iteration (which converges to $V^*$ under standard conditions), **FVI with function approximation does not guarantee convergence** to the optimal value function.

This is an instance of the **deadly triad** (Sutton & Barto):

| Component | Present in FVI? |
|---|-|
| Function approximation | ($\theta^\top \phi(s)$) |
| Bootstrapping | (Bellman backup uses current $\theta$) |
| Off-policy data (sampled states) | |

When all three are present, oscillation or divergence is possible. DQN addresses this with **experience replay** and a **target network** — both are direct engineering fixes for FVI's instability.


## 5. Real-Time Policy Deployment

After $\theta$ converges, we have $V(s) \approx \theta^\top \phi(s)$. The optimal policy is implicitly defined:

$$\pi(s) = \arg\max_a \left[ R(s) + \gamma \cdot \theta^\top \phi(f_{\text{det}}(s, a)) \right]$$

where $f_{\text{det}}$ is the **noise-free** (deterministic) simulator.

**Why remove noise at deployment?**  
- Monte Carlo sampling during training was necessary to learn a robust $V(s)$
- At inference (e.g., 10 Hz control loop), stochastic rollouts add variance without benefit
- The robustness is already baked into $\theta$; using the mean prediction gives the best point estimate

**Execution at each timestep:**
1. Observe current state $s$
2. For each candidate action $a$ (discretized control outputs):  
   Compute $s' = f_{\text{det}}(s, a)$, evaluate $R(s) + \gamma\,\theta^\top\phi(s')$
3. Execute $a^* = \arg\max_a[\cdots]$


## 6. Limitations & Connections to Modern MBRL

### FVI as a Foundation

FVI is the conceptual ancestor of modern deep RL:

| FVI Component | Modern Equivalent |
|---|---|
| Linear $\theta^\top\phi(s)$ | Deep neural network $f_\theta(s)$ |
| Hand-crafted $\phi(s)$ | End-to-end learned representations |
| Tabular action space | Continuous actor (TD3, SAC) |
| Fixed sampled states | Replay buffer (DQN, SAC) |
| Single dynamics model | Ensemble of probabilistic models (PETS, MBPO) |
