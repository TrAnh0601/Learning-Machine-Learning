import numpy as np


def riccati_recursion(A, B, Q, U, T):
    """
    Backward Riccati recursion to compute optimal feedback gains.

    Args:
        A : (n, n)  state transition matrix
        B : (n, d)  control input matrix
        Q : (n, n)  state cost matrix (PSD)
        U : (d, d)  control cost matrix (PD)
        T : int     horizon length

    Returns:
        L : list of T arrays, L[t] is (d, n) feedback gain at time t
        Phi : list of T+1 arrays, Phi[t] is (n, n) value function matrix
    """
    n = A.shape[0]

    Phi = [None] * (T + 1)
    L = [None] * T

    # Base case
    Phi[T] = -Q

    for t in range(T-1, -1, -1):
        P = Phi[t + 1]
        BtP = B.T @ P
        BtPB = BtP @ B
        BtPA = BtP @ A

        # Feedback gain: L_t = -(B^T Phi_{t+1} B - U)^{-1} B^T Phi_{t+1} A
        M = BtPB - U
        L[t] = -np.linalg.solve(M, BtPA)

        # Riccati update: Phi_t = -Q + A^T Phi_{t+1} A - A^T Phi_{t+1} B (B^T Phi_{t+1} B - U)^{-1} B^T Phi_{t+1} A
        Phi[t] = -Q + A.T @ P @ A - BtPA.T @ L[t]

    return L, Phi


def rollout(A, B, L, s0, Sigma, T, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    n = A.shape[0]
    states = np.zeros((T + 1, n))
    actions = np.zeros((T, B.shape[1]))

    states[0] = s0

    for t in range(T):
        a = L[t] @ states[t]
        eps = rng.multivariate_normal(np.zeros(n), Sigma)
        states[t + 1] = A @ states[t] + B @ a + eps
        actions[t] = a

    return states, actions


def compute_total_reward(states, actions, Q, U):
    T = actions.shape[0]
    total = 0.0
    for t in range(T):
        s, a = states[t], actions[t]
        total += -(s @ Q @ s) - (a @ U @ a)
    return total


def demo():
    rng = np.random.default_rng(42)

    # System: 2D state (e.g., position + velocity), 1D control
    n, d = 2, 1
    T    = 20

    A = np.array([[1.0, 1.0],
                  [0.0, 1.0]])       # discrete-time double integrator
    B = np.array([[0.0],
                  [1.0]])            # control affects velocity only

    Q = np.eye(n)                    # equal penalty on position and velocity
    U = np.array([[0.1]])            # low control cost → aggressive corrections

    Sigma = 0.1 * np.eye(n)

    s0 = np.array([5.0, 0.0])       # start far from origin, at rest

    # Run LQR
    L, Phi = riccati_recursion(A, B, Q, U, T)
    states, actions = rollout(A, B, L, s0, Sigma, T, rng=rng)
    total_reward = compute_total_reward(states, actions, Q, U)

    print("=" * 55)
    print("LQR Demo — Double Integrator")
    print("=" * 55)
    print(f"Horizon T        : {T}")
    print(f"Initial state s0 : {s0}")
    print(f"Total reward     : {total_reward:.4f}")
    print()
    print(f"{'t':<5} {'s_t[0] (pos)':<18} {'s_t[1] (vel)':<18} {'a_t'}")
    print("-" * 55)
    for t in range(T + 1):
        a_str = f"{actions[t, 0]:.4f}" if t < T else "—"
        print(f"{t:<5} {states[t, 0]:<18.4f} {states[t, 1]:<18.4f} {a_str}")

    # Phi_t sanity check: should be negative semi-definite throughout
    print()
    print("Phi_t eigenvalue check (all <= 0 expected):")
    for t in [0, T // 2, T]:
        eigvals = np.linalg.eigvalsh(Phi[t])
        print(f"  Phi[{t:>2}] eigenvalues: {eigvals.round(4)}")


if __name__ == "__main__":
    demo()