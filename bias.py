import torch
import pandas as pd
import numpy as np
weights = torch.load(r'weights\yolov3-spp-ultralytics-512.pt')
# str = "Conv2d.weight"
# str1 = 'Conv2d.bias'
str2 = 'BatchNorm2d.bias'

model = weights['model']
# for k, v in model.items():
#     print(k)
# print("--------------------")
module_list3 = []
for k, v in model.items():
    if (str2 in k) == True:
        # print(k)
        key_num = k.split('.')[1]
        key_str = k.split('.')[2]
        key_str1 = k.split('.')[3]
        # print(key_num)
        v = v.view(v.size(0), -1)  # 将四维张量转换为二维张量
        a = [v.cpu().detach().numpy()]  # 将a定义为列表
        for x in np.nditer(a):
            # print(x) # 打印一维的权重值
            module_list3.append(x)
        Data_frame = pd.DataFrame(columns=None, data=module_list3)
        # e.to_csv('base_weights.csv')
        Data_frame.to_csv(f'Wresult/BN_bias/{key_str}_{key_num}_{key_str1}.csv')


