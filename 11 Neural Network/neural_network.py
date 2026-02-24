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
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    @staticmethod
    def sigmoid_derivative(x):
        s = np.array(Activations.sigmoid(x))
        return s * (1 - s)


class Losses:
    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def mse_derivative(y_true, y_pred):
        # dL/d(y_pred) = 2*(y_pred - y_true)/n  [NOT y_true - y_pred]
        return 2 * (y_pred - y_true) / y_true.shape[0]


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.bias = np.zeros((1, output_size))

    def forward(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.bias
        return self.output

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.input.T, output_gradient)
        input_gradient = np.dot(output_gradient, self.weights.T)

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * np.sum(output_gradient, axis=0, keepdims=True)

        return input_gradient


class Activation(Layer):
    def __init__(self, activation, activation_derivative):
        super().__init__()
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, input):
        self.input = input
        self.output = self.activation(input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.activation_derivative(self.input)


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_derivative = None

    def add(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss, loss_derivative):
        self.loss = loss
        self.loss_derivative = loss_derivative

    def predict(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, X, y, epochs=1000, learning_rate=0.1, verbose=False):
        for epoch in range(epochs):
            output = self.predict(X)
            error = self.loss(y, output)

            gradient = self.loss_derivative(y, output)
            for layer in reversed(self.layers):
                gradient = layer.backward(gradient, learning_rate)

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
        }
        """
        nn = NeuralNetwork()

        activations = {
            'relu': (Activations.relu, Activations.relu_derivative),
            'sigmoid': (Activations.sigmoid, Activations.sigmoid_derivative),
            'linear': (lambda x: x, lambda x: np.ones_like(x)),
        }

        losses = {
            'mse': (Losses.mse, Losses.mse_derivative),
        }

        input_size = config['input_size']

        for layer_config in config['layers']:
            layer_type = layer_config['type']

            if layer_type == 'dense':
                units = layer_config['units']
                nn.add(Dense(input_size, units))
                input_size = units

                if 'activation' in layer_config:
                    act_name = layer_config['activation']
                    act_func, act_deriv = activations[act_name]
                    nn.add(Activation(act_func, act_deriv))

        loss_name = config.get('loss', 'mse')
        loss_func, loss_deriv = losses[loss_name]
        nn.set_loss(loss_func, loss_deriv)

        return nn


if __name__ == '__main__':
    np.random.seed(42)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    config = {
        'input_size': 2,
        'layers': [
            {'type': 'dense', 'units': 4, 'activation': 'relu'},
            {'type': 'dense', 'units': 1, 'activation': 'sigmoid'},
        ],
        'loss': 'mse'
    }

    nn = NetworkBuilder.build(config)
    nn.train(X, y, epochs=10000, learning_rate=0.1)

    predictions = nn.predict(X)
    print("Predictions:")
    print(predictions.round(3))
    print("\nTrue values:")
    print(y)