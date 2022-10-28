import torch
import pandas as pd
import numpy as np
weights = torch.load(r'weights\base\yolov3spp-29.pt')
str = "Conv2d.weight"
model = weights['model']
# for k, v in model.items():
#     print(k)
module_list3 = []
for k, v in model.items():
    if (str in k) == True:
        key_num = k.split('.')[1]
        # print(key_num)
        v = v.view(v.size(0), -1)  # 将四维张量转换为二维张量
        a = [v.cpu().detach().numpy()]  # 将a定义为列表
        for x in np.nditer(a):
            # print(x) # 打印一维的权重值
            module_list3.append(x)
        Data_frame = pd.DataFrame(columns=None, data=module_list3)
        # e.to_csv('base_weights.csv')
        Data_frame.to_csv(f'G:/weight_result/first/first_Conv2d{key_num}.csv')