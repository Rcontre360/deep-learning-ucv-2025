import numpy as np
from abc import ABC, abstractmethod

class ActivationFunction(ABC):
    def __init__(self, n:int):
        self.n = n

    @abstractmethod
    def init_sample(self, n:int) -> float:
        pass

    @abstractmethod
    def backward(self, x: np.ndarray) -> np.ndarray:
        pass

# since the best initialization depends on the activation function. We delegate that to its class
class ReLU(ActivationFunction):
    def __call__(self,x:np.ndarray):
        return x.clip(0)

    def init_sample(self):
        return np.random.normal(loc=0, scale=2 / self.n)

    def backward(self,x:np.ndarray):
        return x.clip(0)

