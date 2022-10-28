import torch
import pandas as pd
import numpy as np

weights = torch.load(r'weights\yolov3-spp-ultralytics-512.pt')
# for k, v in weights.items():
#     print(k)
# file_path = 'read_test/ 12.txt'
"""
for k, v in weights.items():
    print(k)
结果如下：
model  optimizer   training_results  epoch  best_map
接下来 取出model
"""
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
        # print(v,v.shape) # 输出变换维度后的二维张量以及二维张量的形状
        a = [v.cpu().detach().numpy()]  # 将a定义为列表
        # print(a)  # 输出numpy格式的数据，二维的
        for x in np.nditer(a):
            # print(x) # 打印一维的权重值
            module_list3.append(x)
        Data_frame = pd.DataFrame(columns=None, data=module_list3)
        # e.to_csv('base_weights.csv')
        Data_frame.to_csv(f'Wresult/Ytrain/Ytrain_Conv2d{key_num}.csv')
        # Data_frame.to_excel('weights/YuTrain_Conv2d51_Excel.xlsx', index=False)

        for x in np.nditer(a):
            # print(x) # 打印一维的权重值
            module_list3.append(x)
"""
for k, v in model.items():
    print(k)
module_list.0.Conv2d.weight
module_list.0.BatchNorm2d.weight
module_list.0.BatchNorm2d.bias
module_list.0.BatchNorm2d.running_mean
module_list.0.BatchNorm2d.running_var
module_list.0.BatchNorm2d.num_batches_tracked
module_list.1.Conv2d.weight
module_list.1.BatchNorm2d.weight
module_list.1.BatchNorm2d.bias
module_list.1.BatchNorm2d.running_mean
module_list.1.BatchNorm2d.running_var
module_list.1.BatchNorm2d.num_batches_tracked
"""
