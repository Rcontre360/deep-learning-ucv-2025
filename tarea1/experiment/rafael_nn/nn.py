import numpy as np

from rafael_nn.layer import Linear
from rafael_nn.optimizer import Optimizer

class NeuralNetwork:
    def __init__(self, layers:list[Linear], optimizer:Optimizer):
        self.layers = layers
        self.optimizer = optimizer

    def forward(self, x: np.ndarray) -> np.ndarray:
        current = x
        for layer in self.layers:
            current = layer.forward(current)

        return current

    def backward(self, loss_grad: np.ndarray) -> None:
        pass

    def update(self, optimizer: Optimizer) -> None:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, loss_fn, optimizer: Optimizer) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
