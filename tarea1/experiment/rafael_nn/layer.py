from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from numpy.typing import NDArray # Import NDArray from numpy.typing

from rafael_nn.acfn import ActivationFunction
from rafael_nn.common import FloatArr
from rafael_nn.optimizer import Optimizer

class Layer(ABC):
    weights: NDArray[np.float64]
    biases: NDArray[np.float64]
    fn: ActivationFunction

    called: bool
    f: FloatArr
    h: FloatArr

    @abstractmethod
    def __call__(self, input: FloatArr) -> tuple[FloatArr,FloatArr]:
        """Compute the loss value."""
        pass

    @abstractmethod
    def backward(self, prediction: FloatArr, target: FloatArr) -> FloatArr:
        """Compute the gradient of the loss with respect to the prediction."""
        pass


class Linear(Layer):
    def __init__(self,prev:int, neurons:int, fn:ActivationFunction):
        """Initializes linear layer with weights. Initializes biases with 0"""
        weights = [[fn.init_sample() for _ in range(prev)] for _ in range(neurons)]
        biases = [0 for _ in range(neurons)]

        self.weights = np.array(weights, dtype=np.float64)
        self.biases = np.array(biases, dtype=np.float64)
        self.fn = fn

    def __call__(self, input: NDArray[np.float64]) -> tuple[FloatArr, FloatArr]:
        """Applies the forward pass. Multiplies the given vector by the matrix weights and applies the activation fn"""
        self.called = True
        # we need the preactivation for the gradient calc
        prev = self.biases + self.weights @ input
        return self.fn(prev), prev

    def backward(self, prev_dl_f:Optional[FloatArr] = None, weights_dl_f: Optional[FloatArr] = None) -> tuple[FloatArr,FloatArr,FloatArr]:
        """
        Calculates dl_weights, dl_f and dl_bias given arguments. weights_dl_f or prev_dl_f must be passed (one or the other).
        if prev_dl_f is passed, this means this is the LAST layer and the first to calculate dl_f. 
        if weights_dl_f is passed, it means this is a hidden layer
        """

        if not self.called:
            raise Exception("Layer must be called before getting derivatives")

        if prev_dl_f is None:
            dl_f = np.where(self.f > 0, 1, 0)*weights_dl_f
        else:
            dl_f = prev_dl_f

        # dl_bias, dl_f and dl_weight
        return dl_f, dl_f, np.matmul(dl_f, self.h.T)



    def update_params(self, optimizer: Optimizer, layer_index: int) -> None:
        pass

