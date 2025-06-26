import numpy as np
from numpy.typing import NDArray
from rafael_nn.common import FloatArr

from rafael_nn.layer import Layer
from rafael_nn.lossfn import LossFunction
from rafael_nn.optimizer import Optimizer

class NeuralNetwork:
    # I like adding types. Its easier to know that what im doing will work, also easier to debug
    # want to update this to use functional programming
    layers: list[Layer]
    optimizer:Optimizer
    layers_output: list[NDArray[np.float64]]
    loss_fn:LossFunction

    def __init__(self, layers:list[Layer], optimizer:Optimizer, loss_fn:LossFunction):
        self.layers = layers
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x)[0]

    def train(self, x: np.ndarray, target:np.ndarray, epochs = 1000, err = 1e-4):
        # here we determine WHEN to stop but the optimizer tells us the next step
        final, all_h, all_f = self._forward(x, True)
        loss = self.loss_fn(final,target)

        if loss < err:
            return

        _,_,dl_w = self._backward(final, target)
        # self.optimizer(self.weights, dl_w)

    # this is almos the same implementation as the 7_2 notebook
    def _forward(self, x: np.ndarray) -> tuple[FloatArr, list[FloatArr], list[FloatArr]]:
        all_h, all_f = [], []
        for i in range(len(self.layers)):
            layer = self.layers[i]
            h, f = layer(x if i == 0 else all_h[i-1])

            all_h.append(h)
            all_f.append(f)

        return all_h[-1], all_h, all_f

    def _backward(self, prediction:FloatArr, target:FloatArr) -> tuple[FloatArr,FloatArr,FloatArr]:
        dl_b, prev_dl_f, dl_w = self.layers[-1].backward(self.loss_fn.backward(prediction,target))

        for i in range(len(self.layers) - 1,-1,-1):
            prev_w = self.layers[i+1].weights.T
            dl_b, prev_dl_f, dl_w = self.layers[i].backward(prev_w.T @ prev_dl_f)

        return dl_b, prev_dl_f, dl_w

    def update(self, optimizer: Optimizer) -> None:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, loss_fn, optimizer: Optimizer) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
