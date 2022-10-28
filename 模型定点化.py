import torch

# 第一步 ：加载模型
content = torch.load('clip_x_32_28_w32_32/yolov3spp-29.pt')
# print("content的key有:", content.keys())  # ['model', 'optimizer', 'training_results', 'epoch', 'best_map']
# 查看模型中有什么东西
model_content = content['model']
"""
['module_list.0.Conv2d.weight',
 'module_list.0.BatchNorm2d.weight', 
 'module_list.0.BatchNorm2d.bias',
'module_list.0.BatchNorm2d.running_mean', 
'module_list.0.BatchNorm2d.running_var', 
'module_list.0.BatchNorm2d.num_batches_tracked'
]
"""
# print(model_content.keys())
# 取'module_list.0.Conv2d.weight'来查看其具体参数
module_0_conv2d_w = model_content["module_list.0.Conv2d.weight"]
# 查看其shape
print("module_list.0.Conv2d.weight这一层的shape是：", module_0_conv2d_w.shape)  # torch.Size([32, 3, 3, 3])
# 查看其参数有多少
num = module_0_conv2d_w.numel()  # 32x3x3x3 = 864个参数
print("module_list.0.Conv2d.weight这一层参数为", num, '这么多。')
print("module_list.0.Conv2d.weight这一层的参数占的字节数为：", num / 1024)

# 做量化
from qatcnn import fake_quantize

module_0_conv2d_w_fake_quantize = fake_quantize(module_0_conv2d_w, 8, 4)
# print(module_0_conv2d_w)
# print(module_0_conv2d_w_quantize)

# 观看具体[0][0]的数字
print(module_0_conv2d_w[0][0])
print("--------------------------------------------------")
print(module_0_conv2d_w_fake_quantize[0][0])
print("**************************************************--------------")
print(module_0_conv2d_w[0][0]*32)
print("\n")

print(module_0_conv2d_w_fake_quantize[0][0]*32)

