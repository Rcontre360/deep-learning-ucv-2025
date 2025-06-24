import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
    @abstractmethod
    def __call__(self, weights: np.ndarray, gradients: np.ndarray, layer_index: int) -> np.ndarray:
        pass

class GradientDescent(Optimizer):
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def __call__(self, parameters: list[np.ndarray], gradients: list[np.ndarray]) -> None:
        for param, grad in zip(parameters, gradients):
            param -= self.learning_rate * grad

