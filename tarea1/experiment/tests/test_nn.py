
import numpy as np
import unittest
from unittest.mock import MagicMock, patch
from typing import List

from rafael_nn.nn import NeuralNetwork

class ActivationFunction:
    """Mock for ActivationFunction."""
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x
    def init_sample(self, i: int, j: int) -> float:
        return 0.0

class Linear:
    """Mock for the Linear layer."""
    def __init__(self, prev: int, neurons: int, fn: ActivationFunction):
        self.prev = prev
        self.neurons = neurons
        self.fn = fn
        self.weights = np.zeros((neurons, prev)) 

    def forward(self, input: np.ndarray) -> np.ndarray:
        return input + 1

class Optimizer:
    """Mock for the Optimizer."""
    def __init__(self):
        pass 
    def optimize(self, params):
        pass

class TestNeuralNetwork(unittest.TestCase):

    def setUp(self):
        """Set up common variables and mocks for tests."""
        self.mock_layer1 = Linear(0,0,ActivationFunction())
        self.mock_layer2 = Linear(0,0,ActivationFunction())
        self.mock_layer3 = Linear(0,0,ActivationFunction())

        self.mock_optimizer = MagicMock(spec=Optimizer)

        self.neural_network = NeuralNetwork(
            layers=[self.mock_layer1, self.mock_layer2, self.mock_layer3],
            optimizer=self.mock_optimizer
        )

    def test_forward_pass_flow_and_output(self):
        initial_input = np.array([5.0, 5.0], dtype=np.float64)

        expected_l1 = initial_input + 1.0
        expected_l2 = expected_l1 + 1.0
        expected_l3 = expected_l2 + 1.0

        output = self.neural_network.forward(initial_input)

        np.testing.assert_array_equal(output, expected_l3)

    def test_forward_pass_empty_layers(self):
        """Test forward pass with no layers."""
        nn_empty = NeuralNetwork(layers=[], optimizer=self.mock_optimizer)
        initial_input = np.array([1.0, 2.0])
        output = nn_empty.forward(initial_input)
        np.testing.assert_array_almost_equal(output, initial_input) # Should return input as is

    def test_optimizer_assignment(self):
        """Test that the optimizer is correctly assigned."""
        self.assertEqual(self.neural_network.optimizer, self.mock_optimizer)

# To run the tests:
if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)

