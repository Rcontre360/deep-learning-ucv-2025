import numpy as np
from numpy.typing import NDArray # Import NDArray from numpy.typing

from rafael_nn.acfn import ActivationFunction
from rafael_nn.common import FloatArr
from rafael_nn.optimizer import Optimizer

class Linear:
    # think that float32 would also do the trick
    weights: NDArray[np.float64]
    biases: NDArray[np.float64]
    fn: ActivationFunction

    def __init__(self,prev:int, neurons:int, fn:ActivationFunction):
        """Initializes linear layer with weights. Initializes biases with 0"""
        weights = [[fn.init_sample() for _ in range(prev)] for _ in range(neurons)]
        biases = [0 for _ in range(neurons)]

        self.weights = np.array(weights, dtype=np.float64)
        self.biases = np.array(biases, dtype=np.float64)
        self.fn = fn

    def forward(self, input: NDArray[np.float64]) -> np.ndarray:
        """Applies the forward pass. Multiplies the given vector by the matrix weights and applies the activation fn"""
        return self.fn(self.biases + self.weights @ input)

    def backward(self, grad_output: FloatArr) -> FloatArr:
        pass

    def update_params(self, optimizer: Optimizer, layer_index: int) -> None:
        pass

