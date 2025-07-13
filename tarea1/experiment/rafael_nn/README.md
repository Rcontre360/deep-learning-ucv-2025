# Rafael NN - Module Documentation

This document provides detailed documentation for each module in the `rafael_nn` package, including class definitions, method signatures, and implementation details.

## Table of Contents

1. [Common Types (`common.py`)](#common-types-commonpy)
2. [Activation Functions (`acfn.py`)](#activation-functions-acfnpy)
3. [Layers (`layer.py`)](#layers-layerpy)
4. [Loss Functions (`lossfn.py`)](#loss-functions-lossfnpy)
5. [Optimizers (`optimizer.py`)](#optimizers-optimizerpy)
6. [Neural Network (`nn.py`)](#neural-network-nnpy)

---

## Common Types (`common.py`)

This module defines common type aliases used throughout the package to ensure type consistency and reduce code verbosity.

```python
import numpy as np
from numpy.typing import NDArray
from typing import TypeAlias

FloatArr: TypeAlias = NDArray[np.float64]
```

### Type Definitions

- **`FloatArr`**: Type alias for `NDArray[np.float64]` - represents a NumPy array of 64-bit floating point numbers. This is the primary data type used for all numerical computations in the package.

---

## Activation Functions (`acfn.py`)

This module implements activation functions used in neural network layers, following an abstract base class pattern for extensibility.

### Abstract Base Class

#### `ActivationFunction(ABC)`

Abstract base class that defines the interface for all activation functions.

**Constructor:**
```python
def __init__(self, n: int)
```
- **Parameters:**
  - `n` (int): Number of neurons in the layer (used for weight initialization)

**Abstract Methods:**

```python
@abstractmethod
def __call__(self, x: FloatArr) -> FloatArr
```
- **Purpose:** Execute the activation function
- **Parameters:** 
  - `x` (FloatArr): Input array to apply activation to
- **Returns:** FloatArr with activation applied

```python
@abstractmethod
def init_sample(self) -> np.float64
```
- **Purpose:** Generate weight initialization values based on the activation function
- **Returns:** Single float64 value for weight initialization

### Concrete Implementations

#### `ReLU(ActivationFunction)`

Rectified Linear Unit activation function implementation.

**Methods:**

```python
def __call__(self, x: FloatArr) -> FloatArr
```
- **Implementation:** `return x.clip(0)`
- **Description:** Applies ReLU activation (max(0, x)) element-wise

```python
def init_sample(self) -> np.float64
```
- **Implementation:** `return np.float64(np.random.normal(loc=0, scale=2 / self.n))`
- **Description:** Uses He initialization (suitable for ReLU) with normal distribution

---

## Layers (`layer.py`)

This module implements neural network layers with a focus on linear (fully connected) layers.

### Abstract Base Class

#### `Layer(ABC)`

Abstract base class defining the interface for all neural network layers.

**Attributes:**
- `weights` (NDArray[np.float64]): Weight matrix for the layer
- `biases` (NDArray[np.float64]): Bias vector for the layer
- `fn` (ActivationFunction): Activation function for the layer
- `f` (FloatArr): Pre-activation values (stored during forward pass)
- `h` (FloatArr): Input values received by the layer (stored during forward pass)

**Abstract Methods:**

```python
@abstractmethod
def __call__(self, input: FloatArr) -> tuple[FloatArr, FloatArr]
```
- **Purpose:** Forward pass through the layer
- **Returns:** Tuple of (activated_output, pre_activation_values)

```python
@abstractmethod
def backward(self, prev_dl_f: Optional[FloatArr] = None, 
             weights_dl_f: Optional[FloatArr] = None) -> tuple[FloatArr, FloatArr, FloatArr]
```
- **Purpose:** Backward pass for gradient computation
- **Returns:** Tuple of (dl_bias, dl_f, dl_weights)

### Concrete Implementations

#### `Linear(Layer)`

Fully connected (dense) linear layer implementation.

**Constructor:**
```python
def __init__(self, prev: int, neurons: int, fn: Optional[ActivationFunction] = None)
```
- **Parameters:**
  - `prev` (int): Number of input features/neurons from previous layer
  - `neurons` (int): Number of neurons in this layer
  - `fn` (Optional[ActivationFunction]): Activation function (defaults to ReLU)

**Key Implementation Details:**
- Weights initialized using the activation function's `init_sample()` method
- Biases initialized to zero with proper shape `(neurons, 1)`
- Stores input (`h`) and pre-activation (`f`) values for backpropagation

**Methods:**

```python
def __call__(self, input: FloatArr) -> tuple[FloatArr, FloatArr]
```
- **Implementation:** 
  - Stores input as `self.h`
  - Computes `self.f = self.biases + self.weights @ input`
  - Returns `(self.fn(self.f), self.f)`

```python
def backward(self, prev_dl_f: Optional[FloatArr] = None, 
             weights_dl_f: Optional[FloatArr] = None) -> tuple[FloatArr, FloatArr, FloatArr]
```
- **Logic:**
  - If `prev_dl_f` is provided: This is the output layer, use it directly
  - If `weights_dl_f` is provided: This is a hidden layer, apply ReLU derivative
  - Computes bias gradients by summing across batch dimension
  - Returns gradients for bias, activation, and weights

**Design Notes:**
- The layer stores both input (`h`) and pre-activation (`f`) values to handle the design challenge where each layer needs access to the previous layer's activation
- Weight initialization is delegated to the activation function for optimal performance

---

## Loss Functions (`lossfn.py`)

This module implements loss functions for training neural networks.

### Abstract Base Class

#### `LossFunction(ABC)`

Abstract base class for all loss functions.

**Abstract Methods:**

```python
@abstractmethod
def __call__(self, prediction: FloatArr, target: FloatArr) -> float
```
- **Purpose:** Compute the loss value
- **Returns:** Scalar loss value

```python
@abstractmethod
def backward(self, prediction: FloatArr, target: FloatArr) -> FloatArr
```
- **Purpose:** Compute gradient of loss with respect to predictions
- **Returns:** Gradient array with same shape as predictions

### Concrete Implementations

#### `MeanSquaredError(LossFunction)`

Mean Squared Error loss function, commonly used for regression tasks.

**Methods:**

```python
def __call__(self, prediction: FloatArr, target: FloatArr) -> np.floating[Any]
```
- **Implementation:** `return np.mean((prediction - target) ** 2)`
- **Description:** Computes average of squared differences

```python
def backward(self, prediction: FloatArr, target: FloatArr) -> FloatArr
```
- **Implementation:** `return 2 * (prediction - target) / prediction.size`
- **Description:** Computes gradient: ∂MSE/∂pred = 2(pred - target) / n

---

## Optimizers (`optimizer.py`)

This module implements optimization algorithms for training neural networks.

### Abstract Base Class

#### `Optimizer(ABC)`

Abstract base class defining the interface for all optimizers.

**Abstract Methods:**

```python
@abstractmethod
def __call__(self, parameters: FloatArr, gradients: FloatArr) -> FloatArr
```
- **Purpose:** Update parameters given gradients
- **Returns:** Updated parameters

```python
@abstractmethod
def load_data(self, x_full: FloatArr, y_full: FloatArr)
```
- **Purpose:** Load and prepare training data

```python
@abstractmethod
def get_batch(self) -> tuple[FloatArr, FloatArr]
```
- **Purpose:** Get next batch of training data
- **Returns:** Tuple of (input_batch, target_batch)

### Concrete Implementations

#### `GradientDescent(Optimizer)`

Standard gradient descent optimizer that uses the entire dataset for each update.

**Constructor:**
```python
def __init__(self, learning_rate: float = 0.01)
```

**Methods:**

```python
def __call__(self, parameters: FloatArr, gradients: FloatArr) -> FloatArr
```
- **Implementation:** `return parameters - self.learning_rate * gradients`
- **Description:** Standard gradient descent update rule

```python
def load_data(self, x_full: FloatArr, y_full: FloatArr)
```
- **Description:** Stores the complete dataset

```python
def get_batch(self) -> tuple[FloatArr, FloatArr]
```
- **Description:** Returns the entire dataset (batch gradient descent)

#### `StochasticGradientDescend(Optimizer)`

Stochastic gradient descent with mini-batch support.

**Constructor:**
```python
def __init__(self, learning_rate: float = 0.01, batch_size: int = 50)
```

**Attributes:**
- `batch` (int): Current batch index (-1 indicates need to reset)
- `batch_size` (int): Size of each mini-batch

**Methods:**

```python
def __call__(self, parameters: FloatArr, gradients: FloatArr) -> FloatArr
```
- **Implementation:** Same as GradientDescent
- **Description:** SGD uses same update rule, difference is in batch management

```python
def load_data(self, x_full: FloatArr, y_full: FloatArr)
```
- **Description:** Shuffles data using random permutation before storing

```python
def get_batch(self) -> tuple[FloatArr, FloatArr]
```
- **Logic:**
  - Resets batch index when end of data is reached
  - Increments batch counter
  - Returns slice of data corresponding to current batch
- **Returns:** `(x[0:1, start_idx:end_idx], y[0:1, start_idx:end_idx])`

**Implementation Note:** The batch indexing uses `x[0:1, start_idx:end_idx]` which suggests the data format expects the first dimension to be preserved while slicing the second dimension for batching.

---

## Neural Network (`nn.py`)

This module contains the main neural network class that orchestrates training and inference.

### Main Class

#### `NeuralNetwork`

Main neural network class that combines layers, optimizers, and loss functions.

**Constructor:**
```python
def __init__(self, layers: list[Layer], optimizer: Optimizer, loss_fn: LossFunction)
```

**Attributes:**
- `layers` (list[Layer]): List of network layers
- `optimizer` (Optimizer): Optimization algorithm
- `loss_fn` (LossFunction): Loss function for training

**Public Methods:**

```python
def __call__(self, x: np.ndarray) -> np.ndarray
```
- **Purpose:** Forward pass for inference
- **Returns:** Network output predictions

```python
def train(self, x_full: FloatArr, y_full: FloatArr, epochs: int = 1000, err: float = 1e-4)
```
- **Purpose:** Train the neural network
- **Parameters:**
  - `x_full` (FloatArr): Complete training input data
  - `y_full` (FloatArr): Complete training target data
  - `epochs` (int): Maximum number of training epochs
  - `err` (float): Error threshold for early stopping
- **Training Loop:**
  1. Load data into optimizer
  2. For each epoch:
     - Get batch from optimizer
     - Forward pass
     - Compute loss
     - Backward pass
     - Update parameters
     - Check stopping criteria

```python
def backward(self, prediction: FloatArr, target: FloatArr) -> tuple[list[FloatArr], list[FloatArr]]
```
- **Purpose:** Public interface for backward pass
- **Returns:** Tuple of (bias_gradients, weight_gradients) for all layers

**Private Methods:**

```python
def _forward(self, x: FloatArr) -> tuple[FloatArr, list[FloatArr], list[FloatArr]]
```
- **Purpose:** Internal forward pass implementation
- **Logic:**
  - Applies activation function to all layers except the last
  - Last layer outputs raw values (no activation)
  - Stores intermediate activations and pre-activations
- **Returns:** Tuple of (final_output, all_activations, all_pre_activations)

```python
def _backward(self, prediction: FloatArr, target: FloatArr) -> tuple[list[FloatArr], list[FloatArr]]
```
- **Purpose:** Internal backward pass implementation
- **Logic:**
  1. Start with loss gradient for output layer
  2. Compute gradients for last layer
  3. Propagate gradients backward through all layers
  4. Each layer computes its own parameter gradients
- **Returns:** Lists of gradients for biases and weights for all layers

**Key Design Features:**

1. **Modular Architecture:** Clear separation between layers, optimizers, and loss functions
2. **Gradient Flow:** Proper gradient computation and propagation through the network
3. **Batch Processing:** Support for both full-batch and mini-batch training
4. **Early Stopping:** Training stops when error threshold is reached
5. **Type Safety:** Comprehensive type hints for better code reliability

**Training Process:**
1. Optimizer manages data batching and shuffling
2. Forward pass computes predictions and stores intermediate values
3. Loss function computes error and its gradient
4. Backward pass propagates gradients through all layers
5. Optimizer updates parameters using computed gradients
6. Process repeats until convergence or epoch limit

The implementation follows a clean object-oriented design with proper separation of concerns, making it easy to extend with new layer types, optimizers, and loss functions.
