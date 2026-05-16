# 🧠 神经网络：从数学底层理解架构

![Python](https://img.shields.io/badge/Python-3.6%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.19%2B-013243)
![License](https://img.shields.io/badge/License-MIT-green)
![状态](https://img.shields.io/badge/Status-Educational-orange)

该神经网络完全由纯'numpy'实现，没有复杂的深度学习架构，只有直观的数学原理

>该方法是用数值微分实现，所以请仅在小型计算或教学上使用，以节约您宝贵的时间

---
## 📖 目录

- [功能](#-features)
- [安装](#-installation)
- [快速开始](#-quick-start)
- [工作原理](#-how-it-works)
- [API 文档](#-api-documentation)
- [示例](#-examples)
- [异或问题](#1-xor-problem)
- [鸢尾花分类](#2-iris-classification)
- [性能说明](#-performance-notes)
- [许可证](#-license)

---

## ✨ 特性

- **零依赖的黑魔法** — 仅依赖 `numpy` ，所有梯度均手动计算。
- **任意深度网络** — 可通过 `net_layer` 自由设置隐藏层数量。
- **数值梯度验证** — 使用中心差分法确保反向传播的正确性，无需推导链式法则。
- **交叉熵 Softmax** — 原生支持多分类任务。
- **轻量且透明** — 大约 120 行代码，非常易读和修改。

---

## 📦 安装

克隆仓库并确保已安装 NumPy：

```bash
git clone https://github.com/linuxlemon256/Neural-Network.git
cd Neural-Network
pip install numpy
```

---

## 🚀 快速开始

```python
import numpy as np
from CustomLayerNet import CustomLayerNet

# 生成模拟数据：100 个样本，5 个特征，3 个类别
X = np.random.randn(100, 5)
y = np.eye(3)[np.random.randint(0, 3, 100)] # 独热标签

# 创建网络：输入 5 → 隐藏层 10 → 输出 3，总共 3 层，训练 500 轮
net = CustomLayerNet(input_size=5, hidden_size=10, output_size=3,
net_layer=3, learning_time=500, learning_rate=0.1)

# 设置训练标准（必需）
net.t = y

# 训练并获得最终预测结果
output = net.function(X)
pred_class = np.argmax(output, axis=1)
print("前 5 个样本的预测类别：", pred_class[:5])
```

---

## ⚙️ 工作原理

### 1. 前向传播

每一层执行线性变换 `X @ W + b`。除了使用 Softmax 的最后一层，所有中间层都使用 Sigmoid 激活函数。

### 2. 反向传播（数值梯度）

对于每个可训练参数，计算该点的损失函数的**中心差分**近似梯度：

grad ≈ (loss(W + h) - loss(W - h)) / (2h)

然后使用标准梯度下降更新参数：

W = W - learning_rate * grad

**注意**：每次迭代需要 `2 × 参数总数` 次前向传播，因此此实现仅适用于小型网络和教学用途。

---
## 📚 API 文档

### `CustomLayerNet` 类

#### 初始化参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `input_size` | `int` | - | 输入特征的数量 |
| `hidden_size` | `int` | - | 所有隐藏层中使用的神经元数量 |
| `output_size` | `int` | - | 输出类别的数量 |
| `net_layer` | `int` | `2` | 网络层总数（包括输入层和输出层） |
| `learning_time` | `int` | `1000` | 总训练迭代次数 |
| `learning_rate` | `float` | `1` | 梯度下降学习率 |
| `training_standard` | `np.ndarray` | `None` | 真实标签的独热编码矩阵，可通过 `net.t` 分配 |
| `batch_size` | `int` | `None` | 保留参数，当前版本未使用 |

#### 核心方法

| 方法 | 返回值 | 描述 |
|------|--------|------|
| `forward(x, fact)` | `np.ndarray` | 执行前向传播并返回Softmax输出概率 |
| `backword(x)` | `list` | 计算数值梯度并更新参数，返回更新后的`self.fact` |
| `function(x)` | `np.ndarray` | 完整的训练过程，返回最终输出概率 |
| `loss(out)` | `float` | 计算交叉熵损失 |
| `softmax(x)` | `np.ndarray` | 按行计算Softmax激活 |
| `sigmoid(x)` | `np.ndarray` | 按元素计算Sigmoid激活 |
---

## 🧪 示例

### 1. XOR 问题

XOR 是一个经典的线性不可分问题，需要至少一个隐藏层才能学习。
```python
import numpy as np
from CustomLayerNet import CustomLayerNet

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[1,0], [0,1], [0,1], [1,0]]) # 两类one-hot

# 创建网络：输入 2 维，隐藏层 4 个神经元，输出 2 维
net = CustomLayerNet(input_size=2, hidden_size=4, output_size=2,
net_layer=3, learning_time=2000, learning_rate=0.1)

# 设置训练标准（真实标签）
net.t = y

# 训练
out = net.function(X)
predicted_classes = np.argmax(out, axis=1)
print("预测的类别序列:", predicted_classes)
```

---

### 2. 鸢尾花分类

使用经典的鸢尾花数据集进行训练和评估。

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from CustomLayerNet import CustomLayerNet
import numpy as np

# 加载数据
iris = load_iris()
X = iris.data
y = OneHotEncoder(sparse=False).fit_transform(iris.target.reshape(-1,1))

# 创建网络：输入4维，隐藏层8个神经元，输出3个类别
net = CustomLayerNet(input_size=4, hidden_size=8, output_size=3,
net_layer=3, learning_time=800, learning_rate=1)

# 设置训练标准
net.t = y

# 训练并获取预测值
pred = net.function(X)
accuracy = np.mean(np.argmax(pred, axis=1) == iris.target)
print(f"鸢尾花数据集上的准确率: {accuracy:.2%}")
```
---
## 🐢 性能描述

由于使用了数值梯度，该实现的计算成本与网络参数的数量成正比。每次训练迭代每个参数都需要两次前向传递（中心差分法）。不同规模网络的近似性能如下：

- **微型网络 (2-4-2)**
例如 XOR 问题，大约有 22 个参数。训练可以在几秒钟内完成。

- **小型网络 (4-8-3)**
例如鸢尾花分类，大约有 67 个参数。训练需要几十秒到大约一分钟。

- **中等网络 (784-128-10)**  
例如，MNIST 手写数字识别，参数超过 100,000 个。每次迭代需要数十万次前向传播，并且单次迭代可能耗时数小时。

**注意**：此方法计算效率低，内存消耗大。

**结论**：此代码库旨在作为教学工具，清楚展示梯度下降的内部机制。对于任何实际规模的任务，该方法在大数据处理上计算能力较低，强烈建议改用基于解析梯度（反向传播算法）的框架，如 PyTorch 或 TensorFlow。

---
## 📄 许可证

本项目遵循 **MIT 许可证**。您可以自由使用、修改和分发该代码。更多详情，请参阅项目根目录中的 [LICENSE](LICENSE) 文件。
