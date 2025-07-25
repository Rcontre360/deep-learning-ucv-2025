import numpy as np
from abc import ABC, abstractmethod

from rafael_nn.common import FloatArr

class Optimizer(ABC):
    @abstractmethod
    def __call__(self, parameters: FloatArr, gradients: FloatArr) -> FloatArr:
        pass

    @abstractmethod
    def load_data(self, x_full: FloatArr, y_full:FloatArr):
        pass

    @abstractmethod
    def get_batch(self) -> tuple[FloatArr, FloatArr]:
        pass


class GradientDescent(Optimizer):
    x: FloatArr
    y: FloatArr

    def __init__(self, learning_rate: float = 0.01):
        """We init this optimizer with the learning rate"""
        self.learning_rate = learning_rate

    def __call__(self, parameters: FloatArr, gradients: FloatArr) -> FloatArr:
        """This is the step value returned"""
        return parameters - self.learning_rate*gradients

    def load_data(self, x_full: FloatArr, y_full:FloatArr):
        """Loads all the data for the class to manage it."""
        self.x = x_full
        self.y = y_full

    def get_batch(self):
        """Gets a batch"""
        return self.x, self.y

class StochasticGradientDescend(Optimizer):
    batch: int
    batch_size: int
    x: FloatArr
    y: FloatArr

    def __init__(self, learning_rate: float = 0.01, batch_size: int = 50):
        """Inits SGD with learning rate and batch size"""
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.batch = -1

    def __call__(self, parameters: FloatArr, gradients: FloatArr) -> FloatArr:
        """Makes a step"""
        return parameters - self.learning_rate*gradients

    def load_data(self, x_full: FloatArr, y_full:FloatArr):
        """Loads the data. Uses performs a permutation of it."""
        perm = np.random.permutation(len(x_full))
        self.x = x_full[perm]
        self.y = y_full[perm]

    def get_batch(self):
        """Obtains the next batch"""
        if (self.batch + 1) * self.batch_size >= len(self.x[0]):
            self.batch = -1

        self.batch+=1
        start_idx, end_idx = self.batch * self.batch_size, (self.batch + 1) * self.batch_size
        return self.x[0:1, start_idx:end_idx], self.y[0:1, start_idx:end_idx]


