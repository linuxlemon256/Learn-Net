# 🧠 Neural-Network

![Python](https://img.shields.io/badge/Python-3.6%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.19%2B-013243)
![License](https://img.shields.io/badge/License-MIT-green)
![状态](https://img.shields.io/badge/Status-Educational-orange)

A multi-layer fully connected neural network implemented purely in NumPy，**Using numerical gradients (central difference) for backpropagation**。No automatic differentiation, no deep learning frameworks—only intuitive mathematical principles。

> Suitable for learning the underlying mechanisms of neural networks, the principles of gradient descent, and hands-on practice, please use only for teaching and learning.。

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

- **Zero-dependency black magic** — Only `numpy` , all gradients are manually computed.
- **Arbitrary deep networks** — Freely set the number of hidden layers through `net_layer`.
- **Numerical gradient verification** — Central difference method ensures the correctness of backpropagation without deriving the chain rule.
- **Cross-entropy Softmax** — Natively supports multi-class tasks.
- **Lightweight and transparent** — About 120 lines of code, very easy to read and modify.

---

## 📦 Install

Clone the repository and make sure NumPy is installed：

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

# Generate simulated data: 100 samples, 5 features, 3 categories
X = np.random.randn(100, 5)
y = np.eye(3)[np.random.randint(0, 3, 100)]   # one-hot label

# Create network: input 5 → hidden layer 10 → output 3, total 3 layers, train for 500 epochs
net = CustomLayerNet(input_size=5, hidden_size=10, output_size=3,
                     net_layer=3, learning_time=500, learning_rate=0.1)

# Set training standards (required)
net.t = y

# Train and obtain the final prediction
output = net.function(X)
pred_class = np.argmax(output, axis=1)
print("Predicted categories of the first 5 samples:", pred_class[:5])
```


---

## ⚙️ Working Principle

### 1. Forward Propagation

Each layer performs a linear transformation `X @ W + b`. Except for the last layer which uses Softmax, intermediate layers all use the Sigmoid activation function.

### 2. Backward Propagation (Numerical Gradient)

For each trainable parameter, compute the **central difference** approximate gradient of the loss function at that point:

grad ≈ (loss(W + h) - loss(W - h)) / (2h)

Then update the parameter using standard gradient descent:

W = W - learning_rate * grad

**Note**: Each iteration requires `2 × total number of parameters` forward passes, so this implementation is only suitable for small networks and educational purposes.

## 📚 API Documentation

### `CustomLayerNet` Class

#### Initialization Parameters

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `input_size` | `int` | - | Number of input features |
| `hidden_size` | `int` | - | Number of neurons used uniformly in all hidden layers |
| `output_size` | `int` | - | Number of output classes |
| `net_layer` | `int` | `2` | Total number of network layers (including input and output layers) |
| `learning_time` | `int` | `1000` | Total number of training iterations |
| `learning_rate` | `float` | `1` | Gradient descent learning rate |
| `training_standard` | `np.ndarray` | `None` | One-hot encoding matrix of the true labels, can be assigned via `net.t` |
| `batch_size` | `int` | `None` | Reserved parameter, not used in the current version |

#### Core Methods

| Method | Return Value | Description |
|------|--------|------|
| `forward(x, fact)` | `np.ndarray` | Performs forward propagation and returns Softmax output probabilities |
| `backword(x)` | `list` | Computes numerical gradients and updates parameters, returning the updated `self.fact` |
| `function(x)` | `np.ndarray` | Full training process, returns the final output probabilities |
| `loss(out)` | `float` | Calculates cross-entropy loss |
| `softmax(x)` | `np.ndarray` | Calculates Softmax activations row-wise |
| `sigmoid(x)` | `np.ndarray` | Calculates Sigmoid activations element-wise |
---
## 🧪 Example

### 1. XOR Problem

XOR is a classic linearly inseparable problem that requires at least one hidden layer to learn.
```python
import numpy as np
from CustomLayerNet import CustomLayerNet

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[1,0], [0,1], [0,1], [1,0]])  # Two-class one-hot encoding

# # Create network: input 2-dimensional, hidden layer with 4 neurons, output 2-dimensional
net = CustomLayerNet(input_size=2, hidden_size=4, output_size=2,
                     net_layer=3, learning_time=2000, learning_rate=0.1)

# Set training standards (true labels)
net.t = y

# Training
out = net.function(X)
predicted_classes = np.argmax(out, axis=1)
print("Predicted category sequence:", predicted_classes)
```

---


### 2. Iris Flower Classification

Train and evaluate using the classic Iris dataset.

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from CustomLayerNet import CustomLayerNet
import numpy as np

# Load data
iris = load_iris()
X = iris.data
y = OneHotEncoder(sparse=False).fit_transform(iris.target.reshape(-1,1))

# Create network: input 4 dimensions, hidden layer with 8 neurons, output 3 classes
net = CustomLayerNet(input_size=4, hidden_size=8, output_size=3,
                     net_layer=3, learning_time=800, learning_rate=1)

# Set Training Standards
net.t = y

# Train and Get Predictions
pred = net.function(X)
accuracy = np.mean(np.argmax(pred, axis=1) == iris.target)
print(f"Accuracy on the Iris dataset: {accuracy:.2%}")
```


---

## 🐢 Performance Description

Since numerical gradients are used, the computational cost of this implementation is proportional to the number of network parameters. Each training iteration requires two forward passes per parameter (central difference method). The approximate performance for networks of different sizes is as follows:

- **Tiny Network (2-4-2)**
For example, the XOR problem, with around 22 parameters. Training can be completed in a few seconds.

- **Small Network (4-8-3)**
For example, Iris classification, with around 67 parameters. Training takes tens of seconds to about a minute.

- **Medium Network (784-128-10)**
For example, MNIST handwritten digit recognition, with more than 100,000 parameters. Each iteration requires hundreds of thousands of forward passes, and a single iteration can take hours.

**Note**: This method has low computational efficiency and high memory usage.

**Conclusion**: This codebase is intended as an educational tool to clearly demonstrate the internal mechanisms of gradient descent. For any practical-scale tasks, this method has low computing ability for large data, and it is strongly recommended to switch to frameworks based on analytical gradients (backpropagation algorithms), such as PyTorch or TensorFlow.

---
## 📄 License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute the code. For more details, please see the [LICENSE](LICENSE) file in the root directory of the project.
