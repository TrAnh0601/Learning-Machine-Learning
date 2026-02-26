"""
neural_network_torch.py — PyTorch rewrite of nn_from_scratch.py

Key differences from the NumPy version:
  - Autograd replaces manual backprop: no backward() methods, no _dW/_db.
    PyTorch builds a computational graph during forward(); .backward() traverses
    it automatically via reverse-mode autodiff.
  - Optimizers are built-in (torch.optim). No need for manual state dicts.
  - nn.Module replaces the custom Layer base class. Parameters registered via
    nn.Parameter are automatically tracked by autograd and the optimizer.
  - Data must be torch.Tensor. dtype=torch.float32 is the standard for neural nets
    (float64 is slower on GPU and unnecessary for most tasks).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Losses:
    mse = nn.MSELoss()
    bce = nn.BCELoss()
    bce_logits = nn.BCEWithLogitsLoss()
    categorical_ce = nn.CrossEntropyLoss()


class Dense(nn.Module):
    def __init__(self, input_size, output_size, l2=0.0):
        super().__init__()
        # nn.Parameter registers tensor as a learnable parameter.
        # autograd tracks all operations on it automatically.
        self.weights = nn.Parameter(torch.empty(input_size, output_size))
        self.bias = nn.Parameter(torch.zeros(1, output_size))
        self.l2 = l2

        # He init: std = sqrt(2 / fan_in)
        nn.init.kaiming_normal_(self.weights, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        output = x @ self.weights + self.bias

        if self.l2 > 0:
            # l2_loss is added to the main loss in NeuralNetwork.train()
            self._l2_penalty = self.l2 * (self.weights ** 2).sum()
        else:
            self._l2_penalty = None
        return output


class Dropout(nn.Module):
    def __init__(self, keep_prob=0.8):
        super().__init__()
        # CONVENTION NOTE: nn.Dropout(p) drops with probability p.
        # My NumPy version used keep_prob. Here: p = 1 - keep_prob.
        self.dropout = nn.Dropout(p=1.0 - keep_prob)

    def forward(self, x):
        return self.dropout(x)


class SoftmaxCELayer(nn.Module):
    def forward(self, x):
        if not self.training:
            return torch.softmax(x, dim=1)
        return x


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        self.loss_fn = None
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            return self.forward(x).numpy()

    def train_model(self, X, y, epochs=1000, verbose=False):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32)

        self.train()

        for epoch in range(epochs):
            # 1. Zero gradients — PyTorch accumulates gradients by default.
            self.optimizer.zero_grad()

            # 2. Forward pass
            output = self.forward(X)

            # 3. Compute loss
            loss = self.loss_fn(output, y)

            # 4. Add L2 penalties from Dense layers
            for layer in self.layers:
                if isinstance(layer, Dense) and layer._l2_penalty is not None:
                    loss = loss + layer._l2_penalty

            # 5. Backward pass
            loss.backward()

            # 6. Optimizer step - updates all parameters using computed gradients
            self.optimizer.step()

            if verbose:
                print(f"Epoch: {epoch+1}/{epochs}: loss={loss.item()}")


class NetworkBuilder:
    @staticmethod
    def build(config):
        """
        config = {
            'input_size': 2,
            'layers': [
                {'type': 'dense',   'units': 64, 'activation': 'relu', 'l2': 1e-4},
                {'type': 'dropout', 'keep_prob': 0.8},
                {'type': 'dense',   'units': 1,  'activation': 'sigmoid'},
            ],
            'loss':      'bce',
            'optimizer': {'type': 'adam', 'learning_rate': 0.001},
        }

        Activation note:
          'softmax_ce' → no Softmax layer; use CrossEntropyLoss which handles it.
          'sigmoid'    → pair with BCELoss (or BCEWithLogitsLoss for logits).
          'linear'     → no activation (regression output).
        """
        # Activation functions — PyTorch functional equivalents
        activations = {
            'relu':    nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh':    nn.Tanh(),
            'linear':  nn.Identity(),
        }

        losses = {
            'mse':            nn.MSELoss(),
            'bce':            nn.BCELoss(),
            'bce_logits':     nn.BCEWithLogitsLoss(),
            'categorical_ce': nn.CrossEntropyLoss(),
        }

        opt_cfg  = config.get('optimizer', {'type': 'adam', 'learning_rate': 0.001})
        opt_type = opt_cfg.get('type', 'adam').lower()
        lr       = opt_cfg.get('learning_rate', 0.001)

        network     = NeuralNetwork()
        input_size  = config['input_size']

        for lc in config['layers']:
            ltype = lc['type']

            if ltype == 'dense':
                units = lc['units']
                act   = lc.get('activation', 'linear')
                l2    = lc.get('l2', 0.0)

                network.add(Dense(input_size, units, l2=l2))

                if act == 'softmax_ce':
                    network.add(SoftmaxCELayer())
                elif act != 'linear':
                    network.add(activations[act])

                input_size = units

            elif ltype == 'dropout':
                network.add(Dropout(keep_prob=lc.get('keep_prob', 0.8)))

        loss_fn = losses[config.get('loss', 'mse')]
        network.set_loss(loss_fn)

        # Optimizer is built AFTER all layers are added so .parameters()
        # captures all registered nn.Parameters.
        # weight_decay here is PyTorch's built-in L2 — alternative to per-layer l2.
        if opt_type == 'sgd':
            optimizer = optim.SGD(network.parameters(), lr=lr,
                                  momentum=opt_cfg.get('momentum', 0.0),
                                  weight_decay=opt_cfg.get('weight_decay', 0.0))
        elif opt_type == 'momentum':
            optimizer = optim.SGD(network.parameters(), lr=lr,
                                  momentum=opt_cfg.get('beta', 0.9),
                                  weight_decay=opt_cfg.get('weight_decay', 0.0))
        elif opt_type == 'adam':
            optimizer = optim.Adam(network.parameters(), lr=lr,
                                   betas=(opt_cfg.get('beta1', 0.9),
                                          opt_cfg.get('beta2', 0.999)),
                                   weight_decay=opt_cfg.get('weight_decay', 0.0))
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")

        network.set_optimizer(optimizer)
        return network


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    config = {
        'input_size': 2,
        'layers': [
            {'type': 'dense',   'units': 8,  'activation': 'relu', 'l2': 1e-4},
            {'type': 'dropout', 'keep_prob': 0.9},
            {'type': 'dense',   'units': 1,  'activation': 'sigmoid'},
        ],
        'loss':      'bce',
        'optimizer': {'type': 'adam', 'learning_rate': 0.01},
    }

    nn_model = NetworkBuilder.build(config)
    nn_model.train_model(X, y, epochs=2000, verbose=False)

    preds = nn_model.predict(X)
    acc   = np.mean((preds > 0.5).astype(int) == y)
    print(f"XOR accuracy: {acc:.2f}")
    print(preds.round(3))
