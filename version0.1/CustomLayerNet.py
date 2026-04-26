import numpy as np

class CustomLayerNet:
    def __init__(self,input_size,hidden_size,output_size,net_layer=2,learning_time=1000,learning_rate=1,training_standard=None,batch_size=None):
        self.lr = learning_rate
        self.t = training_standard
        self.bs = batch_size
        self.lt = learning_time
        self.layer = net_layer
        self.fact = []
        self.grad = []
        self.fact.append([np.random.randn(input_size, hidden_size)*0.1,np.zeros(hidden_size)])
        self.grad.append([[[0 for a in range(hidden_size)] for b in range(input_size)],[0 for c in range(hidden_size)]])
        for _ in range(net_layer-2):
            self.fact.append([np.random.randn(hidden_size, hidden_size)*0.1,np.zeros(hidden_size)])
            self.grad.append([[[0 for a in range(hidden_size)] for b in range(hidden_size)],[0 for c in range(hidden_size)]])
        self.fact.append([np.random.randn(hidden_size, output_size)*0.1,np.zeros(output_size)])
        self.grad.append([[[0 for a in range(output_size)] for b in range(hidden_size)],[0 for c in range(output_size)]])


    def sigmoid(self,x):
        return 1/(np.exp(-x)+1)


    def predict(self,x,fact):
        w,b=fact[0],fact[1]
        out=np.dot(x,w)+b
        return out


    def softmax(self, x):
        h=np.max(x,axis=1,keepdims=True)
        return np.exp(x-h) / np.sum(np.exp(x-h),axis=1,keepdims=True)


    def giant(self, f , x):
        h=1e-8
        return f(x+h) - f (x-h) / (2 * h)

    def loss(self, out):
        eps = 1e-12
        out = np.clip(out, eps, 1 - eps)
        if self.t.ndim == 1:
            N = out.shape[0]
            one_hot = np.zeros_like(out)
            one_hot[np.arange(N), self.t] = 1.0
            target = one_hot
        else:
            target = self.t
        loss = -np.sum(target * np.log(out)) / out.shape[0]
        return loss

    def accuracy(self, out):
        if self.t.ndim == 1:
            true = self.t
        else:
            true = np.argmax(self.t, axis=1)
        pred = np.argmax(out, axis=1)
        return np.mean(pred == true)


    def forward(self, x, fact):
        for i in range(self.layer):
            x = self.predict(x, fact[i])
            if i <self.layer-1:
                x = self.sigmoid(x)
            else:
                x = self.softmax(x)
        return x


    def backward(self, x):
        h=1e-4
        grad=self.grad

        for line in range(len(self.fact)):
            for row in range(len(self.fact[line])):
                if self.fact[line][row].ndim==1:
                    for line1 in range(self.fact[line][row].shape[0]):
                        fact1 = copy.deepcopy(self.fact)
                        fact2 = copy.deepcopy(self.fact)
                        fact1[line][row][line1] += h
                        fx1=self.loss(self.forward(x,fact1))
                        fact2[line][row][line1] -= h
                        fx2 = self.loss(self.forward(x, fact2))
                        grad[line][row][line1]= (fx1 - fx2) / (2 * h)
                else:
                    for line1 in range(self.fact[line][row].shape[0]):
                        for line2 in range(self.fact[line][row].shape[1]):
                            fact1 = copy.deepcopy(self.fact)
                            fact2 = copy.deepcopy(self.fact)
                            fact1[line][row][line1][line2] += h
                            fx1=self.loss(self.forward(x,fact1))
                            fact2[line][row][line1][line2] -= h
                            fx2 = self.loss(self.forward(x, fact2))
                            grad[line][row][line1][line2]= (fx1 - fx2) / (2 * h)
        for line in range(len(self.fact)):
            for row in range(len(self.fact[line])):
                if self.fact[line][row].ndim==1:
                    for line1 in range(self.fact[line][row].shape[0]):
                        self.fact[line][row][line1] -= grad[line][row][line1] * self.lr
                else:
                    for line1 in range(self.fact[line][row].shape[0]):
                        for line2 in range(self.fact[line][row].shape[1]):
                            self.fact[line][row][line1][line2] -= grad[line][row][line1][line2] * self.lr

        return self.fact


    def function(self,x):
        org=x
        epoch=100
        for _ in range(self.lt):
            out=self.forward(org, self.fact)
            self.fact=self.backward(org)
            if _%epoch==0:
                print(f"Iteration {_}, Loss: {self.loss(out):.4f}")
        self.out=out
        out=self.forward(org,self.fact)
        return out

