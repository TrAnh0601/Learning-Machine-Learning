import numpy as np
from neural_network import Activations, Losses, Dense, Activation, NeuralNetwork, NetworkBuilder

results = []

def check(name, condition, info=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({info})" if info else ""))
    results.append(condition)


def he(shape):
    return np.random.randn(*shape) * np.sqrt(2 / shape[0])


# ── 1. Activations ────────────────────────────────────────────────────────────
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

check("relu: zero for negative",          np.all(Activations.relu(x[:2]) == 0))
check("relu: identity for positive",      np.allclose(Activations.relu(x[3:]), x[3:]))
check("relu_derivative: 0 at x<0",       np.all(Activations.relu_derivative(x[:2]) == 0))
check("relu_derivative: 1 at x>0",       np.all(Activations.relu_derivative(x[3:]) == 1))
check("sigmoid: 0.5 at x=0",             np.isclose(Activations.sigmoid(np.array([0.0])), 0.5))
check("sigmoid: output in (0,1)",         np.all((Activations.sigmoid(x) > 0) & (Activations.sigmoid(x) < 1)))
check("sigmoid: numerically stable",      np.all(np.isfinite(Activations.sigmoid(np.array([-1000.0, 1000.0])))))

eps = 1e-5
x_fd = np.array([0.0, 1.0, -1.0, 2.0])
numerical  = (Activations.sigmoid(x_fd + eps) - Activations.sigmoid(x_fd - eps)) / (2 * eps)
analytic   = Activations.sigmoid_derivative(x_fd)
check("sigmoid_derivative: finite difference", np.allclose(analytic, numerical, atol=1e-6))


# ── 2. Losses ─────────────────────────────────────────────────────────────────
y_true = np.array([[1.0], [0.0], [1.0]])
y_pred = np.array([[0.9], [0.1], [0.8]])

check("mse: correct value",   np.isclose(Losses.mse(y_true, y_pred), (0.1**2 + 0.1**2 + 0.2**2) / 3))
check("mse: zero at perfect", np.isclose(Losses.mse(y_true, y_true), 0.0))

grad = Losses.mse_derivative(y_true, y_pred)
check("mse_derivative: correct sign", np.all(np.sign(grad) == np.sign(y_pred - y_true)))

num_grad = np.zeros_like(y_pred)
for i in range(y_pred.shape[0]):
    yp, ym = y_pred.copy(), y_pred.copy()
    yp[i] += eps; ym[i] -= eps
    num_grad[i] = (Losses.mse(y_true, yp) - Losses.mse(y_true, ym)) / (2 * eps)
check("mse_derivative: finite difference", np.allclose(grad, num_grad, atol=1e-5))


# ── 3. Dense gradient check ───────────────────────────────────────────────────
np.random.seed(0)
layer = Dense(3, 4)
X_test = np.random.randn(5, 3)
out = layer.forward(X_test)
check("dense forward: output shape", out.shape == (5, 4))

upstream = np.ones((5, 4))
analytic_wg = np.dot(X_test.T, upstream)
num_wg = np.zeros_like(layer.weights)
for i in range(layer.weights.shape[0]):
    for j in range(layer.weights.shape[1]):
        W = layer.weights.copy()
        W[i, j] += eps; op = np.dot(X_test, W) + layer.bias
        W[i, j] -= 2*eps; om = np.dot(X_test, W) + layer.bias
        num_wg[i, j] = np.sum((op - om) / (2 * eps))
rel_err = np.max(np.abs(analytic_wg - num_wg)) / (np.max(np.abs(analytic_wg)) + 1e-8)
check("dense backward: weight gradient", rel_err < 1e-5, f"rel_err={rel_err:.2e}")


# ── 4. XOR ────────────────────────────────────────────────────────────────────
np.random.seed(0)
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_xor = np.array([[0],[1],[1],[0]], dtype=float)

nn = NeuralNetwork()
d1 = Dense(2, 8); d1.weights = he((2, 8))
d2 = Dense(8, 1); d2.weights = he((8, 1))
nn.add(d1); nn.add(Activation(Activations.relu, Activations.relu_derivative))
nn.add(d2); nn.add(Activation(Activations.sigmoid, Activations.sigmoid_derivative))
nn.set_loss(Losses.mse, Losses.mse_derivative)
nn.train(X_xor, y_xor, epochs=5000, learning_rate=0.1)

acc = np.mean((nn.predict(X_xor) > 0.5).astype(int) == y_xor)
check("xor: 100% accuracy", acc == 1.0, f"acc={acc:.2f}")


# ── 5. Linear Regression ──────────────────────────────────────────────────────
np.random.seed(1)
X_reg = np.random.randn(100, 2)
y_reg = 3 * X_reg[:, 0:1] - 2 * X_reg[:, 1:2] + 1

nn_reg = NetworkBuilder.build({
    'input_size': 2,
    'layers': [{'type': 'dense', 'units': 16, 'activation': 'relu'},
               {'type': 'dense', 'units': 1,  'activation': 'linear'}],
    'loss': 'mse'
})
for l in nn_reg.layers:
    if isinstance(l, Dense): l.weights = he(l.weights.shape)

nn_reg.train(X_reg, y_reg, epochs=3000, learning_rate=0.01)
mse = Losses.mse(y_reg, nn_reg.predict(X_reg))
check("regression: MSE < 0.1", mse < 0.1, f"mse={mse:.4f}")


# ── 6. Multi-output Regression ────────────────────────────────────────────────
np.random.seed(2)
X_mo = np.random.randn(80, 3)
y_mo = np.hstack([X_mo[:, 0:1] + X_mo[:, 1:2], X_mo[:, 1:2] - X_mo[:, 2:3]])

nn_mo = NetworkBuilder.build({
    'input_size': 3,
    'layers': [{'type': 'dense', 'units': 16, 'activation': 'relu'},
               {'type': 'dense', 'units': 2,  'activation': 'linear'}],
    'loss': 'mse'
})
for l in nn_mo.layers:
    if isinstance(l, Dense): l.weights = he(l.weights.shape)

nn_mo.train(X_mo, y_mo, epochs=3000, learning_rate=0.01)
preds_mo = nn_mo.predict(X_mo)
check("multi-output: MSE < 0.1",    Losses.mse(y_mo, preds_mo) < 0.1, f"mse={Losses.mse(y_mo, preds_mo):.4f}")
check("multi-output: output shape", preds_mo.shape == (80, 2))


# ── 7. Deep Network (5 hidden layers) ────────────────────────────────────────
np.random.seed(3)
X_deep = np.random.randn(50, 4)
y_deep = (np.sum(X_deep, axis=1, keepdims=True) > 0).astype(float)

nn_deep = NetworkBuilder.build({
    'input_size': 4,
    'layers': [{'type': 'dense', 'units': 32, 'activation': 'relu'},
               {'type': 'dense', 'units': 16, 'activation': 'relu'},
               {'type': 'dense', 'units': 8,  'activation': 'relu'},
               {'type': 'dense', 'units': 4,  'activation': 'relu'},
               {'type': 'dense', 'units': 1,  'activation': 'sigmoid'}],
    'loss': 'mse'
})
for l in nn_deep.layers:
    if isinstance(l, Dense): l.weights = he(l.weights.shape)

loss_log = []
for _ in range(2000):
    out = nn_deep.predict(X_deep)
    loss_log.append(Losses.mse(y_deep, out))
    g = Losses.mse_derivative(y_deep, out)
    for l in reversed(nn_deep.layers):
        g = l.backward(g, 0.01)

check("deep: loss decreases", loss_log[-1] < loss_log[0], f"{loss_log[0]:.4f} -> {loss_log[-1]:.4f}")
acc_deep = np.mean((nn_deep.predict(X_deep) > 0.5).astype(int) == y_deep)
check("deep: accuracy > 80%", acc_deep > 0.8, f"acc={acc_deep:.2f}")


# ── 8. Memorization ───────────────────────────────────────────────────────────
np.random.seed(4)
X_mem = np.random.randn(10, 2)
y_mem = np.random.randint(0, 2, (10, 1)).astype(float)

nn_mem = NeuralNetwork()
d1 = Dense(2, 32); d1.weights = he((2, 32))
d2 = Dense(32, 1); d2.weights = he((32, 1))
nn_mem.add(d1); nn_mem.add(Activation(Activations.relu, Activations.relu_derivative))
nn_mem.add(d2); nn_mem.add(Activation(Activations.sigmoid, Activations.sigmoid_derivative))
nn_mem.set_loss(Losses.mse, Losses.mse_derivative)
nn_mem.train(X_mem, y_mem, epochs=10000, learning_rate=0.1)

mem_loss = Losses.mse(y_mem, nn_mem.predict(X_mem))
check("memorization: MSE < 0.01", mem_loss < 0.01, f"mse={mem_loss:.6f}")


# ── Summary ───────────────────────────────────────────────────────────────────
passed = sum(results)
print(f"\n{passed}/{len(results)} passed")