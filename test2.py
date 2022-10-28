import torch
import pandas as pd
import numpy as np

"""
如果不把张量数据存储到cpu，在转numpy时会提示：
TypeError: can't convert cuda:0 device type tensor to numpy. 
Use Tensor.cpu() to copy the tensor to host memory first.
"""
total = 0
content = torch.load('weights\base\yolov3spp-29.pt')
file_path = 'read_test/ 12.txt'
model = content['model']
# for k, v in model.items():
#     print(k)
str = "Conv2d.weight"
weights = []
count = 0
for k, v in model.items():
    if (str in k) == True:
        print(k, " :", model[k].shape)
        # print(k, "---", v) # v是张量
        v = v.view(v.size(0), -1)  # 将四维张量转换为二维张量
        # print(v,v.shape) # 输出变换维度后的二维张量以及二维张量的形状
        a = []  # 将a定义为列表
        a.append(v.cpu().numpy())  # 这里先把张量存储到cpu，然后转换为numpy类型
        print(a)  # 输出numpy格式的数据，二维的
        for x in np.nditer(a):
            # print(x) # 打印一维的权重值
            weights.append(x)

        # # np.array(a)
        # # print(np.array(a))
        # b = np.array(a) # 将a转换为array数组
        # c = np.reshape(b, (1, 27)) # 在这里将四维张量转换为二维张量
        e = pd.DataFrame(columns=None, data=weights)
        e.to_csv('weights.csv')
        #
        # count = count +1 # 计算卷积层个数
        # print(count) # 输出卷积层个数

        # w_num = w_num + k.numel()

# str1 = "BatchNorm2d"
# for k,v in model.items():
#     if (str1 in k) == True:
#         print(k, "-----", model[k].shape)
