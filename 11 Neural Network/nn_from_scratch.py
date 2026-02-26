import numpy as np


class Activations:
    @staticmethod
    def relu(x):
        return np.maximum(x, 0)

    @staticmethod
    def relu_derivative(x):
        return x > 0

    @staticmethod
    def sigmoid(x):
        pos = x >= 0
        out = np.empty_like(x, dtype=float)
        out[pos] = 1 / (1 + np.exp(-x[pos]))
        out[~pos] = np.exp(x[~pos]) / (1 + np.exp(x[~pos]))
        return out

    @staticmethod
    def sigmoid_derivative(x):
        s = Activations.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def softmax(x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)


class Losses:
    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def mse_derivative(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.shape[0]

    @staticmethod
    def bce(y_true, y_pred):
        p = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return - np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    @staticmethod
    def bce_derivative(y_true, y_pred):
        p = np.clip(y_pred, 1e-12, 1 - 1e-12)
        m = y_true.shape[0]
        return (-(y_true / p) + (1 - y_true) / (1 - p)) / m

    @staticmethod
    def categorical_ce(y_true, y_pred):
        p = np.clip(y_pred, 1e-12, 1.0)
        return -np.mean(np.sum(y_true * np.log(p), axis=1))

    @staticmethod
    def categorical_ce_derivative(y_true, y_pred):
        # fused Softmax + CE gradient: dL/dz = (y_hat - y_true) / m
        # This is only correct when y_pred already went through softmax
        # and this derivative is applied directly to the pre-softmax z.
        return (y_pred - y_true) / y_true.shape[0]


class Optimizers:
    def update(self, param, gradient, state):
        """Return updated param and new state"""
        raise NotImplementedError


class SGD(Optimizers):
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, param, gradient, state):
        return param - self.lr * gradient, state


class Momentum(Optimizers):
    def __init__(self, learning_rate=0.01, beta=0.9):
        self.lr = learning_rate
        self.beta = beta

    def update(self, param, grad, state):
        v = state.get('v', np.zeros_like(param))
        v = self.beta * v + (1 - self.beta) * grad
        return param - self.lr * v, {'v': v}


class Adam(Optimizers):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def update(self, param, gradient, state):
        m = state.get('m', np.zeros_like(param))
        v = state.get('v', np.zeros_like(param))
        t = state.get('t', 0) + 1

        m = self.beta1 * m + (1 - self.beta1) * gradient
        v = self.beta2 * v + (1 - self.beta2) * (gradient ** 2)

        # bias-corrected estimates
        m_hat = m / (1 - self.beta1 ** t)
        v_hat = v / (1 - self.beta2 ** t)

        param_new = param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return param_new, {'m': m, 'v': v, 't': t}


class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.training = True

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_gradient):
        raise NotImplementedError

    def get_params_and_grads(self):
        """Yield (param_id, param, grad) for all learnable params in this layer."""
        return []


class Dense(Layer):
    def __init__(self, input_size, output_size, l2=0.0):
        super().__init__()
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.bias = np.zeros((1, output_size))
        self.l2 = l2 # L2 regularization coefficient
        self._dw = None
        self._db = None

    def forward(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.bias
        return self.output

    def backward(self, output_gradient):
        m = self.input.shape[0]

        self._dw = np.dot(self.input.T, output_gradient) / m
        self._db = np.sum(output_gradient, axis=0, keepdims=True) / m

        if self.l2 > 0:
            self._dw += (self.l2 / m) * self.weights

        return np.dot(output_gradient, self.weights.T)

    def get_params_and_grads(self):
        yield (id(self), 'w'), self.weights, self._dw
        yield (id(self), 'b'), self.bias, self._db


class Activation(Layer):
    def __init__(self, activation, activation_derivative):
        super().__init__()
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, input):
        self.input = input
        self.output = self.activation(input)
        return self.output

    def backward(self, output_gradient):
        return output_gradient * self.activation_derivative(self.input)

class SoftmaxCELayer(Layer):
    """
    Fused Softmax + Categorical Cross-Entropy output layer.

    Why fuse? Computing dL/dz directly avoids the numerically unstable
    intermediate dL/d(softmax) * d(softmax)/dz computation.
    Result: dL/dz_i = y_hat_i - y_true_i  (see notes §3.1)

    Usage: add as final layer; set loss=Losses.categorical_ce in NeuralNetwork.
    The derivative passed into backward() is the fused gradient, not raw CE grad.
    Use NeuralNetwork.set_loss(Losses.categorical_ce, Losses.categorical_ce_derivative).
    """
    def __init__(self):
        super().__init__()
        self._y_hat = None

    def forward(self, x):
        self.input = x
        self._y_hat = Activations.softmax(x)
        return self._y_hat

    def backward(self, output_gradient):
        # output_gradient here IS the fused dL/dz = (y_hat - y_true)/m
        # passed in from Losses.categorical_ce_derivative
        return output_gradient


class Dropout(Layer):
    """
    Inverted dropout: scale kept activations by 1/p during training so that
    expected value of output is unchanged → no adjustment needed at inference.

    p = keep probability (e.g. p=0.8 drops 20% of units).
    Mask is re-sampled each forward pass (different per batch).
    """
    def __init__(self, keep_prob=0.8):
        super().__init__()
        self.keep_prob = keep_prob
        self._mask = None

    def forward(self, x):
        if not self.training:
            return x                   # no dropout at inference
        self._mask = (np.random.rand(*x.shape) < self.keep_prob) / self.keep_prob
        return x * self._mask

    def backward(self, output_gradient):
        return output_gradient * self._mask if self._mask is not None else output_gradient


class NeuralNetwork:
    def __init__(self, optimizer=None):
        self.layers = []
        self.loss = None
        self.loss_derivative = None
        self.optimizer = optimizer or SGD(learning_rate=0.01)
        self._opt_state = {}

    def add(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss, loss_derivative):
        self.loss = loss
        self.loss_derivative = loss_derivative

    def set_training(self, flag):
        for layer in self.layers:
            layer.training = flag

    def predict(self, input):
        self.set_training(False)
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, X, y, epochs=1000, verbose=False):
        for epoch in range(epochs):
            self.set_training(True)

            # Forward
            output = X
            for layer in self.layers:
                output = layer.forward(output)

            error = self.loss(y, output)

            # Backward
            gradient = self.loss_derivative(y, output)
            for layer in reversed(self.layers):
                gradient = layer.backward(gradient)

            # Optimizer step over all learnable params
            for layer in self.layers:
                for pid, param, g in layer.get_params_and_grads():
                    if g is None:
                        continue
                    state = self._opt_state.get(pid, {})
                    new_param, new_state = self.optimizer.update(param, g, state)
                    param[:] = new_param
                    self._opt_state[pid] = new_state

            if verbose:
                print(f"Epoch {epoch+1}/{epochs}: error: {error:.6f}")


class NetworkBuilder:
    @staticmethod
    def build(config):
        """
        Build a network from a simple configuration

        config = {
            'input_size': 2,
            'layers': [
                {'type': 'dense', 'units': 64, 'activation': 'relu'},
                {'type': 'dense', 'units': 32, 'activation': 'relu'},
                {'type': 'dense', 'units': 1, 'activation': 'sigmoid'}
            ],
            'loss': 'mse'
            'optimizer': {'type': 'SGD', 'learning_rate': 0.001},
        }
        """
        activations = {
            'relu': (Activations.relu, Activations.relu_derivative),
            'sigmoid': (Activations.sigmoid, Activations.sigmoid_derivative),
            'linear': (lambda x: x, lambda x: np.ones_like(x)),
        }

        losses = {
            'mse': (Losses.mse, Losses.mse_derivative),
            'bce': (Losses.bce, Losses.bce_derivative),
            'categorical_ce': (Losses.categorical_ce, Losses.categorical_ce_derivative),
        }

        opt_config = config.get('optimizer', {'type': 'sgd', 'learning_rate': 0.01})
        opt_type = opt_config.get('type', 'sgd').lower()
        lr = opt_config.get('learning_rate', 0.01)

        if opt_type == 'sgd':
            optimizer = SGD(lr)
        elif opt_type == 'momentum':
            optimizer = Momentum(lr, beta=opt_config.get('beta', 0.9))
        elif opt_type == 'adam':
            optimizer = Adam(lr,
                             beta1=opt_config.get('beta1', 0.9),
                             beta2=opt_config.get('beta2', 0.999))
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")

        nn = NeuralNetwork(optimizer=optimizer)
        input_size = config['input_size']

        for layer_config in config['layers']:
            layer_type = layer_config['type']

            if layer_type == 'dense':
                units = layer_config['units']
                act = layer_config.get('activation', 'linear')
                l2 = layer_config.get('l2', 0.0)

                if act == 'softmax_ce':
                    nn.add(Dense(input_size, units, l2=l2))
                    nn.add(SoftmaxCELayer())
                else:
                    nn.add(Dense(input_size, units, l2=l2))
                    act_fn, act_deriv = activations[act]
                    nn.add(Activation(act_fn, act_deriv))
                input_size = units

            elif layer_type == 'dropout':
                nn.add(Dropout(keep_prob=layer_config.get('keep_prob', 0.8)))

        loss_name = config.get('loss', 'mse')
        loss_func, loss_deriv = losses[loss_name]
        nn.set_loss(loss_func, loss_deriv)

        return nn


if __name__ == '__main__':
    np.random.seed(42)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Adam + BCE + Dropout
    config = {
        'input_size': 2,
        'layers': [
            {'type': 'dense', 'units': 8, 'activation': 'relu', 'l2': 1e-4},
            {'type': 'dropout', 'keep_prob': 0.9},
            {'type': 'dense', 'units': 1, 'activation': 'sigmoid'},
        ],
        'loss': 'bce',
        'optimizer': {'type': 'adam', 'learning_rate': 0.01},
    }

    nn = NetworkBuilder.build(config)
    nn.train(X, y, epochs=2000, verbose=False)

    preds = nn.predict(X)
    acc = np.mean((preds > 0.5).astype(int) == y)
    print(f"XOR accuracy: {acc:.2f}")
    print(preds.round(3))