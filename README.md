# 🧠 CustomLayerNet

![Python](https://img.shields.io/badge/Python-3.6%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.19%2B-013243)
![License](https://img.shields.io/badge/License-MIT-green)
![状态](https://img.shields.io/badge/Status-Educational-orange)

一个纯 NumPy 实现的多层全连接神经网络，**使用数值梯度（中心差分）进行反向传播**。没有自动微分，没有深度学习框架——只有直观的数学原理。

> 适合学习神经网络底层机制、梯度下降原理及动手实践,请仅用于教学和学习。

---

## 📖 目录

- [特性](#-特性)
- [安装](#-安装)
- [快速开始](#-快速开始)
- [工作原理](#-工作原理)
- [API 文档](#-api-文档)
- [示例](#-示例)
  - [XOR 问题](#1-xor-问题)
  - [鸢尾花分类](#2-鸢尾花分类)
- [性能说明](#-性能说明)
- [许可证](#-许可证)

---

## ✨ 特性

- **零依赖黑魔法** —— 仅 `numpy` + `copy`（Python 标准库），所有梯度手动计算。
- **任意深度网络** —— 通过 `net_layer` 自由设定隐藏层数量。
- **数值梯度验证** —— 中心差分法确保反向传播的正确性，无需推导链式法则。
- **交叉熵 + Softmax** —— 原生支持多分类任务。
- **轻量透明** —— 代码约 120 行，极易阅读与修改。

---

## 📦 安装

克隆仓库并确保已安装 NumPy：

```bash
git clone https://github.com/yourusername/CustomLayerNet.git
cd CustomLayerNet
pip install numpy
```


---

## 🚀 快速开始

```python
import numpy as np
from CustomLayerNet import CustomLayerNet

# 生成模拟数据：100个样本，5维特征，3个类别
X = np.random.randn(100, 5)
y = np.eye(3)[np.random.randint(0, 3, 100)]   # one-hot 标签

# 创建网络：输入5 → 隐藏层10 → 输出3，共3层，训练500轮
net = CustomLayerNet(input_size=5, hidden_size=10, output_size=3,
                     net_layer=3, learning_time=500, learning_rate=0.1)

# 设置训练标准（必需）
net.t = y

# 训练并获取最终预测
output = net.function(X)
pred_class = np.argmax(output, axis=1)
print("前5个样本的预测类别:", pred_class[:5])
```


---

## ⚙️ 工作原理

### 1.前向传播

每一层执行线性变换 `X @ W + b`，除最后一层使用 Softmax 外，中间层均使用 Sigmoid 激活函数。

### 2.反向传播（数值梯度）

对每个可训练参数，计算损失函数在该点的**中心差分**近似梯度：

grad ≈ (loss(W + h) - loss(W - h)) / (2h)

然后使用普通梯度下降更新参数：

W = W - learning_rate * grad

**注意**：每轮迭代需要执行 `2 × 参数总数` 次前向传播，因此该实现仅适合小型网络与教学场景。

---

## 📚 API 文档

### `CustomLayerNet` 类

#### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_size` | `int` | - | 输入特征的数量 |
| `hidden_size` | `int` | - | 所有隐藏层统一使用的神经元数量 |
| `output_size` | `int` | - | 输出类别数 |
| `net_layer` | `int` | `2` | 网络总层数（包含输入输出层） |
| `learning_time` | `int` | `1000` | 训练迭代总次数 |
| `learning_rate` | `float` | `1` | 梯度下降学习率 |
| `training_standard` | `np.ndarray` | `None` | 真实标签的 one-hot 编码矩阵，可通过 `net.t` 赋值 |
| `batch_size` | `int` | `None` | 预留参数，当前版本未使用 |

#### 核心方法

| 方法 | 返回值 | 说明 |
|------|--------|------|
| `forward(x, fact)` | `np.ndarray` | 执行前向传播，返回 Softmax 输出概率 |
| `backword(x)` | `list` | 计算数值梯度并更新参数，返回更新后的 `self.fact` |
| `function(x)` | `np.ndarray` | 完整训练流程，返回最终输出概率 |
| `loss(out)` | `float` | 计算交叉熵损失值 |
| `softmax(x)` | `np.ndarray` | 按行计算 Softmax 激活值 |
| `sigmoid(x)` | `np.ndarray` | 逐元素计算 Sigmoid 激活值 |
---
## 🧪 示例

### 1. XOR 问题

XOR 是一个经典的线性不可分问题，需要至少一个隐藏层来学习。

```python
import numpy as np
from CustomLayerNet import CustomLayerNet

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[1,0], [0,1], [0,1], [1,0]])  # 两类 one-hot 编码

# 创建网络：输入2维，隐藏层4个神经元，输出2维
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

使用经典的 Iris 数据集进行训练与评估。

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from CustomLayerNet import CustomLayerNet
import numpy as np

# 加载数据
iris = load_iris()
X = iris.data
y = OneHotEncoder(sparse=False).fit_transform(iris.target.reshape(-1,1))

# 创建网络：输入4维，隐藏层8个神经元，输出3类
net = CustomLayerNet(input_size=4, hidden_size=8, output_size=3,
                     net_layer=3, learning_time=800, learning_rate=1)

# 设置训练标准
net.t = y

# 训练并获取预测
pred = net.function(X)
accuracy = np.mean(np.argmax(pred, axis=1) == iris.target)
print(f"在 Iris 数据集上的准确率: {accuracy:.2%}")
```


---

## 🐢 性能说明

由于使用了数值梯度，该实现的计算开销与网络参数的数量成正比。每轮训练需要针对每个参数进行两次前向传播（中心差分法）。以下为不同规模网络的大致性能表现：

- **微型网络 (2-4-2)**  
  例如 XOR 问题，参数数量约为 22 个。训练可在数秒内完成。

- **小型网络 (4-8-3)**  
  例如 Iris 分类，参数数量约为 67 个。训练耗时在数十秒到一分钟左右。

- **中型网络 (784-128-10)**  
  例如 MNIST 手写数字识别，参数数量将超过 10 万个。每轮迭代需要数十万次前向传播，单轮耗时可能长达数小时。

**注意**：该方法的计算能力较低，空间占用大。

**结论**：本代码库旨在作为教学工具，清晰展示梯度下降的内部机制。对于任何实际规模的任务，该方法对于大数据的计算能力低下，强烈建议改用基于解析梯度（反向传播算法）的框架，如 PyTorch 或 TensorFlow。

---
## 📄 许可证

本项目采用 **MIT License**。您可以自由地使用、修改和分发代码，详情请查看项目根目录下的 [LICENSE](LICENSE) 文件。
