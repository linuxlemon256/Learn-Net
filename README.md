# 🧠 Neural Networks: Understanding the Architecture from the Mathematical Foundations

![Python](https://img.shields.io/badge/Python-3.6%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.19%2B-013243)
![License](https://img.shields.io/badge/License-MIT-green)
![状态](https://img.shields.io/badge/Status-Educational-orange)

This neural network is implemented entirely with pure 'numpy', without complex deep learning architectures, only intuitive mathematical principles.
>This method is implemented using mathematical models and principles, so please use it only for small-scale calculations or teaching purposes to save your valuable time.

---
## 📖 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [How It Works](#-how-it-works)
- [API Documentation](#-api-documentation)
- [Examples](#-examples)
- [XOR Problem](#1-xor-problem)
- [Iris Classification](#2-iris-classification)
- [Performance Notes](#-performance-notes)
- [License](#-license)

---
## ✨ Features

- **Zero-dependency black magic** — Only depends on `numpy`, with all gradients computed manually.
- **Arbitrary depth networks** — The number of hidden layers can be freely set via `net_layer`.
- **Numerical gradient verification** — Uses the central difference method to ensure the correctness of backpropagation, without needing to derive the chain rule.
- **Cross-entropy Softmax** — Natively supports multi-class tasks.
- **Lightweight and transparent** — About 350 lines of code, easy to read and modify; removing error prompts reduces it to about 150 lines.
- **Highly flexible** — Many values can be overridden, and of course, direct overriding using `self.xxx` is also possible.
- **Error localization** — Custom bilingual error messages at critical positions, efficiently and accurately pinpointing errors.

---

## 📦 Installation

Clone the repository and make sure NumPy is installed:

```bash
git clone https://github.com/linuxlemon256/Neural-Network.git
cd Neural-Network
pip install numpy
```

---
## 🚀 Quick Start

```python
import numpy as np
from CustomLayerNet import CustomLayerNet

# Generate simulated data: 100 samples, 5 features, 3 classes
X = np.random.randn(100, 5)
y = np.eye(3)[np.random.randint(0, 3, 100)] # One-hot labels

# Create network: input 5 → hidden layer 10 → output 3, total 3 layers, train for 500 epochs
net = CustomLayerNet(input_size=5, hidden_size=10, output_size=3,
net_layer=3, learning_time=500, learning_rate=0.1)

# Train and get final prediction results
output = net.train(X, y)
pred_class = np.argmax(output, axis=1)
print("Predicted classes for the first 5 samples:", pred_class[:5])
```

---
## ⚙️ Working Principle

### 1. Forward Propagation

Each layer performs a linear transformation `X @ W + b`. All intermediate layers use the Sigmoid activation function, except the last layer which uses Softmax.

### 2. Backward Propagation (Numerical Gradient)

For each trainable parameter, compute the **central difference** approximate gradient of the loss function at that point:

grad ≈ (loss(W + h) - loss(W - h)) / (2h)

Then update the parameters using standard gradient descent:

W = W - learning_rate * grad

### 3. Backpropagation of Error

For training data, use derivatives to propagate backward, for example:

dL/dy * dy/dx = dL/dx

Similarly, propagate downstream derivatives to upstream functions in this manner. Only propagating through the loss-softmax function can efficiently compute gradients.

**Note**: Each iteration requires `2 × total number of parameters` forward passes, so this implementation is only suitable for small networks and educational purposes.

---
## 📚 API Documentation

### `CustomLayerNet` Class

#### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|------------|
| `input_size` | `int` | - | Number of input features |
| `hidden_size` | `int` | - | Number of neurons used in all hidden layers |
| `output_size` | `int` | - | Number of output classes |
| `net_layer` | `int` | `2` | Total number of network layers (including input and output layers) |
| `learning_time` | `int` | `1000` | Total number of training iterations |
| `learning_rate` | `float` | `1` | Learning rate for gradient descent |
| `print_every` | `int` | `100` | How often to print loss during `train()` |
| `method` | `dict` | `{"activation":"relu","backpropagation":"errorback","train":"ordinary"}` | Neural network configuration dictionary: `activation` – activation function, options are `"sigmoid"` or `"relu"`; `backpropagation` – backpropagation method, options are `"errorback"` or `"numerical_differentiation"`; `"train"` reserved for future updates |
| `init_weights` | `float` or `None` | `0.1` | Standard deviation for weight initialization; if set to `None`, automatically choose `Xavier (sigmoid)` or `He (relu)` initialization based on the activation function |
| `print_output` | `bool` | `True` | Whether to print loss and other information during training |
---
#### Core Methods

| Method | Return Value | Description |
|------|--------|------|
| `forward(x)` | `np.ndarray` | Performs forward propagation, returns the softmax probability output (shape `(batch_size, output_size)`), and saves activations of each layer in `self.activations` |
| `backword(x)` | `list` | Computes the gradients of the loss with respect to each layer's parameters, returns a list of gradients (same structure as `self.fact`); automatically converts integer labels to `one‑hot` |
| `train(x, t, method=None)` | `np.ndarray` | Trains using full-batch gradient descent for `learning_time` iterations, returns the final `softmax` output for the entire training set; can temporarily override configuration via the `method` parameter |
| `loss(out)` | `float` | Computes cross-entropy loss (automatically converts `self.t` to `one‑hot` to match `out`) |
| `softmax(x)` | `np.ndarray` | Computes softmax activation row-wise (for the output layer) |
| `sigmoid(x)` | `np.ndarray` | Computes sigmoid activation element-wise |
| `accuracy(out, t=None)` | `float` | Computes classification accuracy; can pass `t` as true labels (supports integer or one‑hot), otherwise uses `self.t` |
| `update(grad)` | - | Updates all layer weights and biases based on gradient `grad` and learning rate |
| `relu(x)` | `np.ndarray` | Computes ReLU activation element-wise (`max(0, x)`) |
| `affine(x, fact)` | `np.ndarray` | Affine transformation: `np.dot(x, w) + b` |
---
> **Note**: `backward` automatically converts integer labels to one-hot during backpropagation and uses Leaky ReLU (negative slope `0.1`) to avoid neuron death. `accuracy` supports passing in labels `t` directly without pre-setting `self.t`.
---

## 🧪 Example

### 1. XOR Problem

XOR is a classic linearly inseparable problem that requires at least one hidden layer to learn.
```python
import numpy as np
from CustomLayerNet import CustomLayerNet

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[1,0], [0,1], [0,1], [1,0]]) # two-class one-hot

# Create network: input 2 dimensions, hidden layer with 4 neurons, output 2 dimensions
net = CustomLayerNet(input_size=2, hidden_size=4, output_size=2,
net_layer=3, learning_time=2000, learning_rate=0.1)

# Training
out = net.train(X,y)
predicted_classes = np.argmax(out, axis=1)
print("Predicted class sequence:", predicted_classes)
```

---
### 2. Iris Flower Classification

Using the classic Iris dataset for training and evaluation.

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from CustomLayerNet import CustomLayerNet
import numpy as np

# Load data
iris = load_iris()
X = iris.data
y = OneHotEncoder(sparse=False).fit_transform(iris.target.reshape(-1,1))

# Create network: input size 4, hidden layer with 8 neurons, output 3 classes
net = CustomLayerNet(input_size=4, hidden_size=8, output_size=3,
net_layer=3, learning_time=800, learning_rate=1)

# Train and get predictions
pred = net.train(X,y)
accuracy = np.mean(np.argmax(pred, axis=1) == iris.target)
print(f"Accuracy on the Iris dataset: {accuracy:.2%}")
```
---
## 🐢 Performance Description

Since numerical gradients are used, the computational cost of this implementation is proportional to the number of network parameters. Each parameter requires two forward passes per training iteration (central difference method). Approximate performance for networks of different sizes is as follows:

- **Tiny Network (2-4-2)**
For example, the XOR problem, which has about 22 parameters.
- ***Numerical Differentiation:***
- Training can be completed in a few seconds
- ***Backpropagation:***
- Can be completed in fractions of a second

- **Small Network (4-8-3)**
For example, Iris classification, which has about 67 parameters.
- ***Numerical Differentiation:***
- Training can be completed within a minute
- ***Backpropagation:***
- Can be completed within a few to tens of seconds

- **Medium Network (784-128-10)**
For example, MNIST handwritten digit recognition, with over 100,000 parameters.
- ***Numerical Differentiation:***
- Not tested; each iteration requires hundreds of thousands of forward passes, and a single iteration may take hours.
- ***Backpropagation:***
- Takes several to tens of minutes to complete

**Note**: Numerical differentiation is computationally inefficient and memory-intensive.

**Conclusion**: This codebase is intended as a teaching tool to clearly demonstrate the internal mechanisms of gradient descent. For tasks of any practical scale, this method is computationally weak for large data processing, and it is strongly recommended to use frameworks based on analytical gradients (backpropagation algorithms), such as PyTorch or TensorFlow.

---
## 📄 License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this code. For more details, please refer to the [LICENSE](LICENSE) file in the project's root directory.
