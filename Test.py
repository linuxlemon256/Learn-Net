from CustomLayerNet import *
np.random.seed(42) #选一个真命天子(随机种子)/Choose a random seed

# 数据准备：2个样本，2维输入，2分类/Data preparation: 2 samples, 2-dimensional input, 2 classes
t = np.array([[1, 0],                           # 第一个样本为类0/The first sample belongs to class 0
              [0, 1]])                          # 第二个样本为类1/The second sample belongs to class 1
dt = np.asarray(t)
x = np.random.randn(*dt.shape)                  #生成随机的输入数据/Generate random input data


net = CustomLayerNet(input_size=2,                    # 创建网络/Create a deep learning network
               hidden_size=3,
               output_size=2,
               net_layer=2,
               learning_time=1000,
               learning_rate=5,
               training_standard=t)


first_predict=net.forward(x,net.fact)           #最初预测/Initial prediction
first_loss=net.loss(net.forward(x,net.fact))    #最初损失/Initial loss

print("\n开始训练/training...")
y = net.function(x)
print("训练前预测/Pre-training prediction：")
print(first_predict)
print(f"\n最初损失/Initial loss: {first_loss:.6f}")
print("\n训练后预测/Post-training prediction：")
print(y)
print(f"\n最终损失/Final loss: {net.loss(net.forward(x,net.fact)):.6f}")
