
# Rafael NN

A minimal neural network implementation using NumPy for educational purposes. This package provides a simple and clean implementation of feedforward neural networks with backpropagation, designed for the DL-UCV 1-2025 course.

## Installation

### From PyPI

```bash
pip install rafael_nn

```

### From Source

```bash
git clone <your-repository-url>
cd rafael_nn
pip install -e .

```

### Requirements

-   Python >= 3.8
-   NumPy

## Quick Start

```python
import numpy as np
from rafael_nn import Linear, MeanSquaredError, NeuralNetwork, GradientDescent, StochasticGradientDescend

np.random.seed(0)

def print_weights(nn):
    for layer in nn.layers:
        print(layer)
        print("weight", layer.weights)
        print("bias", layer.biases, "\n")

# Create network architecture
n_layers = 3
n_by_layer = 30
layers = [Linear(1, n_by_layer)] + [Linear(n_by_layer, n_by_layer) for _ in range(n_layers-1)] + [Linear(n_by_layer, 1)]

# Initialize loss function and neural network
loss_fn = MeanSquaredError()
nn = NeuralNetwork(layers, optimizer=StochasticGradientDescend(0.01, 20), loss_fn=loss_fn)

# Print initial weights
print_weights(nn)

# Train your network (add your training data here)
# nn.train(X_train, y_train, epochs=1000)

```

## Project Structure

```
rafael_nn/
├── pyproject.toml          # Project configuration and dependencies
├── README.md              # This file
├── rafael_nn/             # Main package directory
│   ├── __init__.py        # Package initialization
│   ├── nn.py              # Main neural network class
│   ├── layer.py           # Layer implementations (Dense, etc.)
│   ├── acfn.py            # Activation functions (ReLU, Sigmoid, etc.)
│   ├── lossfn.py          # Loss functions (MSE, CrossEntropy, etc.)
│   ├── optimizer.py       # Optimization algorithms (SGD, Adam, etc.)
│   └── common.py          # Common utilities and helper functions
├── tests/                 # Test suite
│   ├── __init__.py
│   ├── test_nn.py         # Neural network tests
│   ├── test_layer.py      # Layer tests
│   ├── test_acfn.py       # Activation function tests
│   ├── test_nn_backprop.py # Backpropagation tests
│   └── test_train.py      # Training tests

```

## Basic Commands

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_nn.py

# Run with verbose output
pytest tests/ -v

```

## Contributing

1.  Fork the repository
2.  Create a feature branch (`git checkout -b feature/amazing-feature`)
3.  Make your changes
4.  Add tests for new functionality
5.  Run the test suite (`pytest tests/`)
6.  Commit your changes (`git commit -m 'Add amazing feature'`)
7.  Push to the branch (`git push origin feature/amazing-feature`)
8.  Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Rafael Contreras**  
Email: rcontreraspimentel@gmail.com

## Acknowledgments

-   Created for the DL-UCV 1-2025 course
-   Built with educational purposes in mind
-   Inspired by popular deep learning frameworks
