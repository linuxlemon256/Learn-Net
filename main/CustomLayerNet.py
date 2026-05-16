import numpy as np

class NNError(Exception):
    """神经网络异常基类 / Base exception for neural network"""
    pass

class ShapeMismatchError(NNError):
    """形状不匹配 / Shape mismatch"""
    pass

class ConfigError(NNError):
    """配置错误 / Configuration error"""
    pass

class MissingTargetError(NNError):
    """标签未设置 / Target labels not set"""
    pass

class GradientError(NNError):
    """梯度计算异常 / Gradient computation error"""
    pass

class CustomLayerNet:
    #构建神经网络/Build a neural network
    def __init__(self, input_size,output_size,hidden_size=3,net_layer=2,learning_time=1000,learning_rate=0.5,print_every=100,init_weights=0.1,
                 method={"activation":"relu","backpropagation":"errorback","train":"ordinary"},print_output=True):
        self.method = method
        if init_weights is None:
            if method["activation"] == "sigmoid":
                init_weights_in=np.sqrt(1.0 / input_size)
                init_weights_out=np.sqrt(1.0 / hidden_size)
            elif method["activation"] == "relu":
                init_weights_in=np.sqrt(2.0 / input_size)
                init_weights_out=np.sqrt(2.0 / hidden_size)
        else:
            init_weights_in=init_weights
            init_weights_out=init_weights
        self.lr = learning_rate
        self.lt = learning_time
        self.t=None
        self.layer = net_layer
        self.pe = print_every
        self.print_output = print_output
        self.fact = []
        self.fact.append([np.random.randn(input_size, hidden_size)*init_weights_in,np.zeros(hidden_size)])
        for _ in range(net_layer-2):
            self.fact.append([np.random.randn(hidden_size, hidden_size)*init_weights_out,np.zeros(hidden_size)])
        self.fact.append([np.random.randn(hidden_size, output_size)*init_weights_out,np.zeros(output_size)])


        #以下是该类的异常,若旨在学习神经网络知识那您可以无视,异常报错类均已使用注释#------划分
        #The following are the exceptions of this class. If your goal is to learn about neural networks, you can ignore them. All exception error classes have been divided using comments #------
        #-------------------------------


        required_keys = {"train","activation", "backpropagation"}       # 验证method字典 / Validate method dictionary
        valid_train={"ordinary","mini_batch"}
        valid_activations = {"sigmoid", "relu"}
        valid_backprops = {"errorback", "numerical_differentiation"}

        if not required_keys.issubset(self.method.keys()):
            raise ConfigError(
                f"method字典必须包含 {required_keys}，当前键为: {list(self.method.keys())} / "
                f"method dict must contain {required_keys}, got keys: {list(self.method.keys())}")

        if self.method["train"] not in valid_train:
            raise ConfigError(
                f"不支持的训练方式: {self.method['train']}，可选: {valid_train} / "
                f"Unsupported train: {self.method['train']}, choose from: {valid_train}")

        if self.method["activation"] not in valid_activations:
            raise ConfigError(
                f"不支持的激活函数: {self.method['activation']}，可选: {valid_activations} / "
                f"Unsupported activation: {self.method['activation']}, choose from: {valid_activations}")


        if self.method["backpropagation"] not in valid_backprops:
            raise ConfigError(
                f"不支持的反向传播方法: {self.method['backpropagation']}，可选: {valid_backprops} / "
                f"Unsupported backpropagation: {self.method['backpropagation']}, choose from: {valid_backprops}")


        if self.layer < 1:                  # 验证网络层数 / Validate number of layers
            raise ConfigError("网络层数必须 >= 1 / Number of layers must be >= 1")
        if len(self.fact) != self.layer:
            raise ConfigError(
                f"权重列表fact长度 {len(self.fact)} 与 self.layer ({self.layer}) 不匹配 / "
                f"Length of fact list {len(self.fact)} does not match self.layer ({self.layer})")

        #-------------------------------

    def sigmoid(self,x):
        out = 1/(np.exp(-x)+1)
        return out

    def relu(self,x):
        out=np.maximum(0, x)
        return out

    def affine(self, x,fact):
        w,b=fact[0],fact[1]
        out=np.dot(x,w)+b
        return out

    def softmax(self, x):
        h=np.max(x,axis=1,keepdims=True)
        return np.exp(x-h) / np.sum(np.exp(x-h),axis=1,keepdims=True)

    def loss(self, out):

        eps = 1e-12
        out = np.clip(out, eps, 1 - eps)
        if self.t.ndim == 1:
            n = out.shape[0]
            one_hot = np.zeros_like(out)
            one_hot[np.arange(n), self.t] = 1.0
            target = one_hot
        else:
            target = self.t
        loss = -np.sum(target * np.log(out)) / out.shape[0]
        return loss

    def accuracy(self, out, t=None):
        if t is None:
            t = self.t
        if t is None:
            raise MissingTargetError(...)

        if t.ndim == 2:
            t = np.argmax(t, axis=1)

        # 如果 out 已经是类别标签（1维），直接比较；否则先 argmax
        if out.ndim == 2:
            pred = np.argmax(out, axis=1)
        else:
            pred = out  # 假设已经是一维预测

        return np.mean(pred == t)

    def update(self,grad):
        for layer_idx in range(self.layer):
            self.fact[layer_idx][0]-=self.lr*grad[layer_idx][0]
            self.fact[layer_idx][1]-=self.lr*grad[layer_idx][1]

    def forward(self, x):

        # ---------------------------------

        if x.ndim != 2:         # 检查输入维度 / Check input dimensions
            raise ShapeMismatchError(
                f"输入必须是2维 (batch_size, features)，实际维度: {x.ndim} / "
                f"Input must be 2D (batch_size, features), got ndim: {x.ndim}")
        if x.shape[0] == 0:
            raise ValueError("批次大小不能为0 / Batch size cannot be zero")


        expected_in_features = self.fact[0][0].shape[0]     # 检查输入特征数与第一层权重匹配 / Check feature dimension matches first layer weights
        if x.shape[1] != expected_in_features:
            raise ShapeMismatchError(
                f"输入特征数 {x.shape[1]} 与第一层权重期望的 {expected_in_features} 不匹配 / "
                f"Input features {x.shape[1]} does not match expected {expected_in_features} for layer 0")

        #以上为异常/The above is abnormal
        # ---------------------------------
        #以下为前向逻辑/The following is forward logic

        self.activations = [x]
        for i in range(self.layer):
            x = self.affine(x, self.fact[i])
            if i <self.layer-1:
                if self.method["activation"] == "sigmoid":
                    x = self.sigmoid(x)
                elif self.method["activation"] == "relu":
                    x=self.relu(x)
                self.activations.append(x)
            else:
                x = self.softmax(x)
                self.activations.append(x)
        return x

    def backward(self, x):

        # ---------------------------------


        if not hasattr(self, 't') or self.t is None:        # 检查标签是否存在 / Check if target labels exist
            raise MissingTargetError(
                "标签 self.t 未设置，无法计算损失梯度 / self.t is not set, cannot compute loss gradient")


        if self.t.ndim == 1:        # 标签形状验证 / Validate label shape
            if x.shape[0] != len(self.t):
                raise ShapeMismatchError(
                    f"输出样本数 {x.shape[0]} 与标签长度 {len(self.t)} 不匹配 / "
                    f"Number of samples {x.shape[0]} != length of labels {len(self.t)}")
        elif self.t.ndim == 2:
            if x.shape[0] != self.t.shape[0]:
                raise ShapeMismatchError(
                    f"输出形状 {x.shape} 与标签形状 {self.t.shape} 不匹配 / "
                    f"Output shape {x.shape} does not match label shape {self.t.shape}")

        # 以上为异常/The above is abnormal
        # ---------------------------------
        # 以下为逻辑/The following is logic

        h=1e-4
        grad=[]

        if self.method["backpropagation"] == "numerical_differentiation":
            for layer_idx in range(len(self.fact)):
                w = self.fact[layer_idx][0]
                b = self.fact[layer_idx][1]
                w_grad = np.zeros_like(self.fact[layer_idx][0])
                b_grad = np.zeros_like(self.fact[layer_idx][1])
                for i in range(w.shape[0]):
                    for j in range(w.shape[1]):
                        w[i, j] += h
                        loss1 = self.loss(self.forward(x))
                        w[i, j] -= 2 * h
                        loss2 = self.loss(self.forward(x))
                        w[i, j] += h
                        w_grad[i, j] = (loss1 - loss2) / (2 * h)
                for k in range(b.shape[0]):
                    b[k] += h
                    loss1 = self.loss(self.forward(x))
                    b[k] -= 2 * h
                    loss2 = self.loss(self.forward(x))
                    b[k] += h
                    b_grad[k] = (loss1 - loss2) / (2 * h)

                grad.append([w_grad, b_grad])
            return grad

        elif self.method["backpropagation"] == "errorback":

            out = self.forward(x)
            batch_size=out.shape[0]
            grad=[]
            if self.t.ndim == 1:
                onehot = np.zeros((batch_size, out.shape[1]), dtype=out.dtype)
                onehot[np.arange(batch_size), self.t] = 1.0
                target = onehot
            else:
                target = self.t
            dout = (out - target) / batch_size


            #-----------------------
            if not hasattr(self, 'activations') or len(
                    self.activations) == 0:  # 检查前向传播是否执行过 / Check if forward has been called
                raise RuntimeError("必须先调用 forward 才能进行反向传播 / Must call forward before backward")
            # -----------------------


            for layer_idx in reversed(range(self.layer)):
                if layer_idx < self.layer-1:
                    if self.method["activation"] == "sigmoid":
                        sigmoid_out=self.activations[layer_idx+1]
                        sigmoid_dout=(sigmoid_out*(1-sigmoid_out))
                        dout=dout*sigmoid_dout
                    elif self.method["activation"] == "relu":
                        relu_out=self.activations[layer_idx+1]
                        relu_dout = np.where(relu_out > 0, 1, 0.1)
                        dout=dout*relu_dout  #relu函数的导数是一,但它的输出可能是零/relu_dout=1,but relu_out may is zero
                x_out=self.activations[layer_idx]
                w = self.fact[layer_idx][0]

                #--------------------------------

                if x_out.shape[1] != w.shape[0]:    # 检查维度对齐 / Check dimension alignment
                    raise ShapeMismatchError(
                        f"第{layer_idx}层：输入特征 {x_out.shape[1]} 与权重矩阵行数 {w.shape[0]} 不匹配 / "
                        f"Layer {layer_idx}: input features {x_out.shape[1]} != weight rows {w.shape[0]}")

                #--------------------------------

                grad_w=np.dot(x_out.T,dout)
                grad_b=np.sum(dout,axis=0)
                if layer_idx >0:
                    w=self.fact[layer_idx][0]
                    dout=np.dot(dout,w.T)
                grad.insert(0,[grad_w, grad_b])
            return grad

    def train(self,x,t,method=None):
        self.t = t
        if method is not None:

            # ------------------------

            required_keys = {"train", "activation", "backpropagation"}  # 验证method字典 / Validate method dictionary
            valid_train = {"ordinary", "mini_batch"}
            valid_activations = {"sigmoid", "relu"}
            valid_backprops = {"errorback", "numerical_differentiation"}

            if not required_keys.issubset(method.keys()):
                raise ConfigError(
                    f"method字典必须包含 {required_keys}，当前键为: {list(method.keys())} / "
                    f"method dict must contain {required_keys}, got keys: {list(method.keys())}")

            if method["train"] not in valid_train:
                raise ConfigError(
                    f"不支持的训练方式: {method['train']}，可选: {valid_train} / "
                    f"Unsupported train: {method['train']}, choose from: {valid_train}")

            if method["activation"] not in valid_activations:
                raise ConfigError(
                    f"不支持的激活函数: {method['activation']}，可选: {valid_activations} / "
                    f"Unsupported activation: {method['activation']}, choose from: {valid_activations}")

            if method["backpropagation"] not in valid_backprops:
                raise ConfigError(
                    f"不支持的反向传播方法: {method['backpropagation']}，可选: {valid_backprops} / "
                    f"Unsupported backpropagation: {method['backpropagation']}, choose from: {valid_backprops}")

            if self.layer < 1:  # 验证网络层数 / Validate number of layers
                raise ConfigError("网络层数必须 >= 1 / Number of layers must be >= 1")
            if len(self.fact) != self.layer:
                raise ConfigError(
                    f"权重列表fact长度 {len(self.fact)} 与 self.layer ({self.layer}) 不匹配 / "
                    f"Length of fact list {len(self.fact)} does not match self.layer ({self.layer})")
            # ------------------------

            self.method = method

        # -----------------------------------


        if x.ndim != 2:             # 检查输入维度 / Check input dimensions
            raise ShapeMismatchError("训练输入 x 必须是2维数组 / Training input x must be a 2D array")
        if self.t.ndim == 1 and x.shape[0] != len(self.t):
            raise ShapeMismatchError(
                f"输入样本数 {x.shape[0]} 与标签数 {len(self.t)} 不匹配 / "
                f"Number of samples {x.shape[0]} != number of labels {len(self.t)}")


        #-----------------------------------

        org=x
        pe=self.pe
        for _ in range(self.lt):
            grad=self.backward(org)
            self.update(grad)
            if _%pe==0 and self.print_output:
                print(f"Iteration {_}, Loss: {self.loss(self.activations[-1]):.8f}")
        return self.forward(org)

    def train_SGD(self, x, t, epochs=10, learning_rate=None, print_every=None, batch_size=32):
        if learning_rate is not None:
            self.lr = learning_rate
        if print_every is not None:
            self.pe = print_every
        self.t = t
        total_batches=int(np.ceil(x.shape[0] / batch_size))
        for epoch in range(epochs):
            n = np.random.permutation(x.shape[0])
            x_shuffle=x[n]
            t_shuffle=t[n] if t.ndim == 1 else t[n]
            for i in range(total_batches):
                start = i * batch_size
                end = min((i+1) * batch_size, x.shape[0])
                x_batch = x_shuffle[start:end]
                t_batch = t_shuffle[start:end]
                self.t=t_batch
                grad=self.backward(x_batch)
                self.update(grad)
                if ((i+1)%self.pe==0 or i==total_batches-1) and self.print_output:
                    out_batch=self.activations[-1]
                    loss=self.loss(out_batch)
                    print(f"time {epoch + 1}/{epochs} ,"
                          f"Batch {i + 1}/{total_batches}, "
                          f"Loss: {loss:.8f}")
        self.t=t
        return self.forward(x)



