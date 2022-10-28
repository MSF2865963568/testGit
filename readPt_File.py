import torch

total = 0
content = torch.load('yolov3spp-29.pt')
print(content)
file_path = 'read_test/ 11.txt'
# for k, v in content.items():
#     print("k:", k)
# model = content['model']
# content = dict(content)
# with open(file_path, 'a') as f:
#     print(content, file=f)
# for k, v in model.items():
#     print(k)

# with open(file_path, 'w') as f:
#     f.write(model)
