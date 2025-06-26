
import unittest
import numpy as np
from rafael_nn.acfn import ReLU
from rafael_nn.layer import Layer, Linear
from rafael_nn.lossfn import MeanSquaredError

from rafael_nn.nn import NeuralNetwork
from rafael_nn.optimizer import GradientDescent


class TestNNGradient(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        layers = 3
        n_by_layer = 5
        layers:list[Layer] = [Linear(1,n_by_layer)] + [Linear(n_by_layer,n_by_layer) for _ in range(layers)] + [Linear(n_by_layer,1)]

        self.nn = NeuralNetwork(layers, optimizer=GradientDescent(), loss_fn=MeanSquaredError())

    def test_nn_gradient_calc(self):
        x = np.random.randn(1)
        target = np.array([1.0])
        epsilon = 1e-5
        tolerance = 1e-4

        prediction, _, _ = self.nn._forward(x)
        dx_analytical, _, _ = self.nn._backward(prediction, target)

        dx_numerical = np.zeros_like(x)
        for i in range(x.size):
            x_pos = x.copy()
            x_neg = x.copy()
            x_pos[i] += epsilon
            x_neg[i] -= epsilon

            fx_pos, _, _ = self.nn._forward(x_pos)
            fx_neg, _, _ = self.nn._forward(x_neg)

            loss_pos = self.nn.loss_fn(fx_pos, target)
            loss_neg = self.nn.loss_fn(fx_neg, target)

            dx_numerical[i] = (loss_pos - loss_neg) / (2 * epsilon)

        # === Compare ===
        assert np.allclose(dx_analytical, dx_numerical, atol=tolerance)
        print("Gradient check passed!")
        pass

