import numpy as np
from rafael_nn.optimizer import Optimizer
from numpy.typing import NDArray # Import NDArray from numpy.typing
from rafael_nn.acfn import ActivationFunction

class Linear:
    weights: NDArray[np.float64]
    fn: ActivationFunction

    def __init__(self,prev:int, neurons:int, fn:ActivationFunction):
        # using this function we initialize the weights for the current layer
        self.weights = np.fromfunction(fn.init_sample,(neurons,prev),dtype=float)
        self.fn = fn

    def forward(self, input: np.ndarray) -> np.ndarray:
        # forward pass. We just multiply the input vector by the matrix weights.R Returns a vector
        return self.fn(self.weights @ input)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        pass

    def update_params(self, optimizer: Optimizer, layer_index: int) -> None:
        pass


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
