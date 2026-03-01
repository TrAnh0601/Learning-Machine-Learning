# Chapter: Markov Decision Processes (MDPs) & Value/Policy Iteration

## 1. Introduction to Markov Decision Processes

A **Markov Decision Process (MDP)** provides a mathematical framework for modeling sequential decision-making under uncertainty. An MDP is formally defined by a 5-tuple $(S, A, P_{sa}, \gamma, R)$:

| Component | Description |
|-----------|-------------|
| $S$ | Finite set of **states** |
| $A$ | Finite set of **actions** |
| $P_{sa}$ | **Transition probability distribution**: $P(s_{t+1} = s' \mid s_t = s, a_t = a)$ |
| $\gamma$ | **Discount factor**: $\gamma \in [0, 1)$, controls importance of future rewards |
| $R$ | **Reward function**: $R : S \to \mathbb{R}$, immediate reward received at each state |

**Goal:** Find a **policy** $\pi : S \to A$ that maximizes the expected discounted total reward:

$$\mathbb{E}\left[R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + \cdots\right]$$


## 2. Value Functions

### 2.1 Policy Value Function $V^\pi$

The value function $V^\pi(s)$ represents the expected discounted total reward starting from state $s$ and following policy $\pi$ forever:

$$V^\pi(s) = \mathbb{E}\left[R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + \cdots \mid s_0 = s,\ \pi\right]$$

### 2.2 Bellman Equation

$V^\pi$ satisfies the following recursive relationship — the value of a state equals the immediate reward plus the discounted expected value of the next state:

$$V^\pi(s) = R(s) + \gamma \sum_{s' \in S} P_{s\pi(s)}(s')\ V^\pi(s')$$

> **Deep Dive — Linear System Interpretation:**  
> For a fixed policy $\pi$, the Bellman equation is a **linear system** of $|S|$ equations in $|S|$ unknowns $\{V^\pi(s)\}_{s \in S}$. This allows exact computation of $V^\pi$ via standard linear algebra (e.g., direct matrix inversion).


## 3. Optimal Value and Optimal Policy

### 3.1 Optimal Value Function $V^*$

$V^*(s)$ is the maximum expected discounted total reward achievable from state $s$:

$$V^*(s) = \max_\pi\ V^\pi(s)$$

### 3.2 Bellman Optimality Equation

The optimal value function satisfies its own Bellman equation, incorporating a maximization over all possible actions:

$$V^*(s) = R(s) + \max_{a \in A}\ \gamma \sum_{s' \in S} P_{sa}(s')\ V^*(s')$$

### 3.3 Recovering the Optimal Policy $\pi^*$

Once $V^*$ is computed, the optimal policy is obtained by acting **greedily** with respect to $V^*$:

$$\pi^*(s) = \arg\max_{a \in A} \sum_{s' \in S} P_{sa}(s')\ V^*(s')$$


## 4. Algorithms for Solving MDPs

### 4.1 Value Iteration

Computes $V^*$ by repeatedly applying the Bellman optimality operator:

1. Initialize $V(s) := 0$ for all $s$.
2. Repeat until convergence:

$$V(s) := R(s) + \max_{a \in A}\ \gamma \sum_{s' \in S} P_{sa}(s')\ V(s')$$

### 4.2 Policy Iteration

Operates directly on the policy rather than iterating on the value function:

1. Initialize $\pi$ randomly.
2. Repeat until convergence:
   - **Policy Evaluation:** Solve the linear system to compute $V^\pi$.
   - **Policy Improvement:** Update the policy greedily:

$$\pi(s) := \arg\max_{a \in A} \sum_{s'} P_{sa}(s')\ V^\pi(s')$$


## 5. Reinforcement Learning in Practice

### 5.1 Estimating Transition Probabilities from Data

In many real-world settings (e.g., robot control), $P_{sa}$ is unknown and must be estimated from observed interactions:

$$\hat{P}_{sa}(s') = \frac{\text{number of transitions from } s \text{ to } s' \text{ via action } a}{\text{total number of times action } a \text{ was taken in state } s}$$

### 5.2 Exploration vs. Exploitation

An agent using a purely greedy policy (always picking the highest-reward action given current estimates) risks getting stuck in local optima. Common mitigation strategies:

- **$\varepsilon$-greedy:** With probability $1 - \varepsilon$, take the current optimal action; with probability $\varepsilon$, take a random action to gather new data.


## 6. Meta-Insights & Summary

**Synchronous vs. Asynchronous Updates:**  
Value Iteration can be implemented synchronously (update all states simultaneously via matrix operations) or asynchronously (sweep state-by-state). Synchronous updates are generally preferred as they leverage GPU parallelism.

**Computational Trade-off — Value vs. Policy Iteration:**  
Policy Iteration typically converges in fewer iterations, but each iteration is more expensive due to solving a linear system at $O(|S|^3)$ cost. Value Iteration is generally more scalable and practical for large state spaces.

**MDP as the Foundation of RL:**  
The MDP framework underpins every modern Reinforcement Learning algorithm. The combination of value function optimization and deliberate exploration enables agents to learn complex behaviors without hand-coded rules.