# encoding=utf-8

import sys
import math
import numpy as np

alpha = 0.001  # 步长

x = np.array([(2, 0, 0), (2, 1, -1), (2, 2, -2), (0, 1, -1), (0, 2, -2), (0, 1, -1)])
w = np.array([0.1, 0.1, 0.1]).reshape([-1, 1])  # 初始随机设置权值

m = len(x)  # 训练数据条数

epoch_num = 100  # 最大迭代次数

for epoch in range(epoch_num):
    print('第%d轮：' % (epoch + 1))
    # 遍历训练数据集，不断更新w值
    z = np.dot(x, w)  # (6,3)*(3,1)=(6,1)
    a = 1 / (1 + np.exp(-z))  # (6,1)
    dz = a - 1  # (6,1)
    dw = np.dot(x.T, dz)  # (3,6)*(6,1)=(3,1)
    w = w - alpha * dw
    print(w)
    C = -a * z + np.log(1 + np.exp(z))
    print('Loss:', np.mean(C))
