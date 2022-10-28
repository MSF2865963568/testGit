import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


content = torch.load('yolov3spp-29.pt')
str = "Conv2d.weight"
weights = []
model = content['model']
count = 0
for k, v in model.items():
    if (str in k) == True:
        # print(k, " :", model[k].shape)
        # print(k, "---", v) # v是张量
        v = v.view(v.size(0), -1) # 将四维张量转换为二维张量
        # print(v,v.shape) # 输出变换维度后的二维张量以及二维张量的形状
        a = []  # 将a定义为列表
        a.append(v.cpu().numpy()) # 这里先把张量存储到cpu，然后转换为numpy类型
        # print(a) # 输出numpy格式的数据，二维的
        for x in np.nditer(a):
            # print(x) # 打印一维的权重值
            weights.append(x)
print(type(weights))
for i in weights:
    print(i)


# a = np.fromfile(open('file', 'r'), sep='\n')
# # [ 0.     0.005  0.124  0.     0.004  0.     0.111  0.112]
#
# # You can set arbitrary bin edges:
# bins = [0, 0.150]
# hist, bin_edges = np.histogram(a, bins=bins)
# # hist: [8]
# # bin_edges: [ 0.    0.15]
#
# # Or, if bin is an integer, you can set the number of bins:
# bins = 4
# hist, bin_edges = np.histogram(a, bins=bins)
# # hist: [5 0 0 3]
# # bin_edges: [ 0.     0.031  0.062  0.093  0.124]
#
