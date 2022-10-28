import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import pandas as pd


# 读取csv数据，并返回均值和方差
def getData(csv_path):
    with open(csv_path) as csv_file:
        row = csv.reader(csv_file, delimiter=',')
        print("row的type:", type(row))
        print(row)

        next(row)  # 读取首行
        price = []  # 建立一个数组来数据

        # 读取除首行之后每一行的第1列数据，并将其加入到数组price之中
        for r in row:
            price.append(float(r[0]))  # 将字符串数据转化为浮点型加入到数组之中
    mean_value = np.var(price)  # 输出均值
    variance = np.mean(price)  # 输出方差
    return variance, mean_value


def gd(data_x, mu=0, sigma=1):
    """根据公式，由自变量x计算因变量的值

    Argument:
        x: array
            输入数据（自变量）
        mu: float
            均值
        sigma: float
            方差
    """
    left = 1 / (np.sqrt(2 * math.pi) * np.sqrt(sigma))
    right = np.exp(-(data_x - mu) ** 2 / (2 * sigma))
    return left * right


if __name__ == '__main__':
    path_data = r'../weights.csv'
    # 自变量
    # 读取csv文件数据读取为numpy.array型数据
    data_array = np.genfromtxt(path_data)[1:]
    mean_data, variance_data = getData(path_data)
    print(mean_data)
    print(variance_data)
    print("---------------------------------")
    #  因变量（不同均值或方差）
    y_1 = gd(data_array, mean_data, variance_data)
#  绘图
plt.plot(data_array, y_1, color='green')
# #  因变量（不同均值或方差）
# y_1 = gd(x, 0, 0.2)
# y_2 = gd(x, 0, 1.0)
# y_3 = gd(x, 0, 5.0)
# y_4 = gd(x, -2, 0.5)
#
# #  绘图
# plt.plot(x, y_1, color='green')
# plt.plot(x, y_2, color='blue')
# plt.plot(x, y_3, color='yellow')
# plt.plot(x, y_4, color='red')
#  设置坐标系
plt.xlim(-5.0, 5.0)
plt.ylim(-0.2, 1)

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
plt.legend(labels=['$\mu =  mean_data, \sigma^2=variance_data$'])
# plt.legend(labels=['$\mu =  mean_data, \sigma^2=variance_data$', '$\mu = 0, \sigma^2=1.0$', '$\mu = 0,
# \sigma^2=5.0$', '$\mu = -2, \sigma^2=0.5$'])
plt.show()
