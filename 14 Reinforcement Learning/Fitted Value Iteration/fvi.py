import numpy as np
import gymnasium as gym


# Features
STATE_LOW = np.array([-2.4, -2.0, -0.21, -2.0])
STATE_HIGH = np.array([ 2.4,  2.0,  0.21,  2.0])

RBF_CENTERS = np.random.default_rng(0).uniform(size=(50, 4))

def normalize(s):
    return (s - STATE_LOW) / (STATE_HIGH - STATE_LOW)

def phi(s):
    """RBF features on normalized state: avoids sigma mismatch across dimensions."""
    diff = RBF_CENTERS - normalize(s)
    return np.exp(-np.sum(diff**2, axis=1) / (2 * 0.2**2))


# Simulator
ENV = gym.make("CartPole-v1")
ENV.reset()

def simulate(s, a, noisy=False):
    """One-step simulation from arbitrary state s."""
    ENV.reset()
    ENV.unwrapped.state = s.copy()
    s_next, r, terminated, truncated, _ = ENV.step(a)
    if noisy:
        s_next += np.random.normal(0, 0.02, size=4)
    return s_next, r, terminated or truncated


# Fitted Value Iteration
def fitted_value_iteration(n_states=500, n_iter=50, k=5, gamma=0.99, ridge=1e-4):
    """
    FVI loop:
      1. Sample m states randomly
      2. For each state, estimate Q(s,a) via k Monte Carlo rollouts
      3. y[i] = max_a Q(s[i], a)
      4. theta <- ridge regression (Phi^T Phi + lambdaI)^{-1} Phi^T y
      5. Repeat until convergence
    """
    rng = np.random.default_rng(42)
    states = rng.uniform(STATE_LOW, STATE_HIGH, (n_states, 4))
    theta = np.zeros(50)

    best_theta = theta.copy()
    best_return  = -np.inf

    for iter in range(n_iter):
        Phi = np.stack([phi(s) for s in states])
        y   = np.zeros(n_states)

        for i, s in enumerate(states):
            best_q = -np.inf
            for a in [0, 1]:
                q = 0.0
                for _ in range(k):
                    s_next, r, done = simulate(np.asarray(s), a, noisy=True)
                    v_next = 0.0 if done else theta @ phi(s_next)
                    q += r + gamma * v_next
                best_q = max(best_q, q / k)
            y[i] = best_q

        # Ridge regression: stabilizes LSTD when Phi^T Phi is ill-conditioned
        A = Phi.T @ Phi + ridge * np.eye(50)
        theta_new = np.linalg.solve(A, Phi.T @ y)

        delta = np.linalg.norm(theta_new - theta)
        theta = theta_new

        current_return = run_policy(theta, n_episodes=20, verbose=False)
        print(f"Iter {iter + 1:2d} | Delta = {delta:.5f} | Eval Return (3 eps) = {current_return:.1f}")

        if current_return > best_return:
            best_return = current_return
            best_theta = theta.copy()

        if delta < 1e-4:
            print("Converged")
            break

        if best_return >= 500:
            print("Find best policy")
            break

    return best_theta


# Policy evaluation
def run_policy(theta, gamma=0.99, n_episodes=100, verbose=False):
    """Greedy policy: pi(s) = argmax_a [R(s) + gamma * theta @ phi(s')]"""
    eval_env = gym.make("CartPole-v1")
    plan_env = gym.make("CartPole-v1")
    plan_env.reset()
    returns = []

    def lookahead(s, a):
        plan_env.reset()
        plan_env.unwrapped.state = s.copy()
        s_next, r, term, trunc, _ = plan_env.step(a)
        done = term or trunc
        return r if done else r + gamma * (theta @ phi(s_next))

    for _ in range(n_episodes):
        s, _ = eval_env.reset()
        total, done = 0.0, False
        while not done:
            a = int(np.argmax([lookahead(s, a) for a in [0, 1]]))
            s, r, term, trunc, _ = eval_env.step(a)
            total += r
            done = term or trunc
        returns.append(total)

    eval_env.close()
    plan_env.close()
    mean, std = float(np.mean(returns)), np.std(returns)
    if verbose:
        print(f"Mean return ({n_episodes} eps) of Best Policy: {mean:.1f} +/- {std:.1f}")
    return mean


# Experiment
if __name__ == "__main__":
    theta = fitted_value_iteration()
    run_policy(theta, verbose=True)
    ENV.close()