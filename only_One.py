import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

content = torch.load('clip_x_32_28_w32_32/yolov3spp-29.pt')
# print("content的key有:", content.keys())  # ['model', 'optimizer', 'training_results', 'epoch', 'best_map']
str_Conv2d = "Conv2d.weight"
model = content['model']  # 获取model
print("model.keys():", model.keys())
lay_count = 0
list_v = []
weight_x = []
for k, v in model.items():
    lay_count = lay_count + 1
    # if (str_Conv2d in k) == True:
    if lay_count == 1 and (str_Conv2d in k) == True:
        print("k的名字：", k, "---", model[k].shape)
        print("这一层中有多少个参数：", model[k].numel())  # 32X3X3X3=864
        print("这一层占多少内存：", 864 * 4 / 1024)  # 864*4--占的字节数，除以1024 = 3.375 k大小（内存）
        v = v.view(v.size(0), -1)  # 将四维张量转换为二维张量tensor
        # print(v, v.shape)
        # print(v)
        print("----------------------")
        list_v.append(v.cpu().numpy())  # 这里先把张量存储到cpu，然后转换为numpy类型:array
        # print(list_v)  # 输出numpy格式的数据，二维的
        # for x in np.nditer(list_v, flags=['external_loop'], order="F"):
        #     print(x, end=" \n")
        list_it = np.nditer(list_v, flags=['f_index'])
        print(list_it.ndim)  # 判断列表是一维的还是二维的
        while not list_it.finished:
            weight_x.append(list_it[0])
            list_it.iternext()
    else:
        break
# for i in wight_x:
#     print(i)
weight_max = max(weight_x)
print("weight_max", weight_max)
weight_min = min(weight_x)
print("weight_min", weight_min)
e = pd.DataFrame(columns=None, data=weight_x)
e.to_csv('32_28_32_32_lay0ne_weight_x.csv')
