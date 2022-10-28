import torch
from matplotlib import pyplot as plt

model = torch.load("./result/base/resNetFpn-model-14.pth", map_location='cpu')
model = model['model']


def plot_weights(model):
    # modules = [module for module in model.modules()]
    num_sub_plot = 0
    for i, layer in enumerate(model):
        # print(i, layer)
        # if hasattr(layer, 'weight'):
        if 'weight' in layer:
            print(layer)
            w = model[layer]  # layer.weight.data
            w = w.reshape(-1)
            print(w.size())
            if i == 0:
                a = w
            else:
                a = torch.cat((a, w), 0)
            # print(a.size())
            # print(type(w))
            # print(w[w<=1])
            '''
            plt.subplot(221 + num_sub_plot)
            w = model[layer] #layer.weight.data
            #print(w)
            w_one_dim = w.cpu().numpy().flatten()
            plt.hist(w_one_dim, bins=50)
            num_sub_plot += 1
            '''
    # w_one_dim = a.cpu().numpy().flatten()
    # plt.hist(w_one_dim, bins=5)
    # plt.show()
    b = abs(a)
    print(b[b <= 1].numel())
    print(b[b <= 0.5].numel())
    print(b[b <= 0.25].numel())
    print(b.numel())
    print(b[b <= 1].numel() / b.numel())
    print(b[b <= 0.5].numel() / b.numel())
    print(b[b <= 0.25].numel() / b.numel())
    print(b[b <= 0.125].numel() / b.numel())
    print(b[b <= 0.0625].numel() / b.numel())
    print(b[b <= 0.03125].numel() / b.numel())
    print(b[b <= 0.015625].numel() / b.numel())


plot_weights(model)
