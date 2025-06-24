import numpy as np
from numpy.typing import NDArray

from rafael_nn.layer import Linear
from rafael_nn.lossfn import LossFunction
from rafael_nn.optimizer import Optimizer

class NeuralNetwork:
    # I like adding types. Its easier to know that what im doing will work, also easier to debug
    # want to update this to use functional programming
    layers: list[Linear]
    optimizer:Optimizer
    layers_output: list[NDArray[np.float64]]
    loss_fn:LossFunction

    def __init__(self, layers:list[Linear], optimizer:Optimizer, loss_fn:LossFunction):
        self.layers_output = [] * len(layers)
        self.layers = layers
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x)

    def train(self, x: np.ndarray):
        # here we determine WHEN to stop but the optimizer tells us the next step
        self._forward(x, True)
        gradients = self._backward()
        self.weights = self.optimizer(self.weights, gradients)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x)

    def _forward(self, x: np.ndarray, train=False) -> np.ndarray:
        current = x
        for i in range(len(self.layers)):
            layer = self.layers[i]
            current = layer.forward(current)
            if train:
                self.layers_output[i] = current

        return current

    def _backward(self) -> None:
        # here the loss fn is used
        pass

    def update(self, optimizer: Optimizer) -> None:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, loss_fn, optimizer: Optimizer) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
