import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

layerWeight = []
layerBias = []


def predict(x):
    z1 = x.dot(layerWeight[0]) + layerBias[0]
    a1 = np.tanh(z1)

    z2 = a1.dot(layerWeight[1]) + layerBias[1]
    a2 = np.tanh(z2)

    z3 = a2.dot(layerWeight[2]) + layerBias[2]
    a3 = np.exp(z3)
    out = a3 / np.sum(a3, axis=1, keepdims=True)

    #返回两个里大的啊(假装概率大的就是呗
    return np.argmax(out, axis=1)


def plot(pred_func, x, y, n):
    # Set min and max values and give it some padding
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(x[:, 0], x[:, 1], s=40, c=y, cmap=plt.cm.Spectral)

    plt.title(str(n) + " iterations")
    plt.pause(0.01)


# 生成数据集
np.random.seed(0)
x, y = datasets.make_moons(500, noise=0.1)
plt.scatter(x[:, 0], x[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
plt.title("dataset")
plt.pause(2)
#plt.show()

# 2个隐含层的神经网络
nnInputDim = 2  # 输入层节点数
nnHidenDim = [5, 6]  # 两个隐含层节点数
nnOutputDim = 2  # 输出层节点数
learningRate = 0.0005  # 学习率

# 初始化
layerWeight.append(np.random.randn(nnInputDim, nnHidenDim[0]))
layerWeight.append(np.random.randn(nnHidenDim[0], nnHidenDim[1]))
layerWeight.append(np.random.randn(nnHidenDim[1], nnOutputDim))

layerBias.append(np.zeros((1, nnHidenDim[0])))
layerBias.append(np.zeros((1, nnHidenDim[1])))
layerBias.append(np.zeros((1, nnOutputDim)))

# 训练
for i in range(0, 10001):
    # 向前传播，两个隐含层使用tanh，输出层使用softmax
    z1 = x.dot(layerWeight[0]) + layerBias[0]
    a1 = np.tanh(z1)

    z2 = a1.dot(layerWeight[1]) + layerBias[1]
    a2 = np.tanh(z2)

    z3 = a2.dot(layerWeight[2]) + layerBias[2]
    a3 = np.exp(z3)
    out = a3 / np.sum(a3, axis=1, keepdims=True)

    # 反向传播
    deltaOut = out
    deltaOut[range(500), y] -= 1
    deltaW3 = a2.T.dot(deltaOut)
    deltaB3 = np.sum(deltaOut, axis=0, keepdims=True)

    delta2 = deltaOut.dot(layerWeight[2].T) * (1 - np.power(a2, 2))
    deltaW2 = a1.T.dot(delta2)
    deltaB2 = np.sum(delta2, axis=0, keepdims=True)

    delta1 = delta2.dot(layerWeight[1].T) * (1 - np.power(a1, 2))
    deltaW1 = x.T.dot(delta1)
    deltaB1 = np.sum(delta1, axis=0, keepdims=True)

    layerWeight[2] -= learningRate * deltaW3
    layerWeight[1] -= learningRate * deltaW2
    layerWeight[0] -= learningRate * deltaW1
    layerBias[2] -= learningRate * deltaB3
    layerBias[1] -= learningRate * deltaB2
    layerBias[0] -= learningRate * deltaB1

    if (i % 1000 == 0 or i % 100 == 0 and i < 500 or i % 10 == 0 and i < 50 or i < 5):
        plot(lambda x: predict(x), x, y, i)

plt.show();