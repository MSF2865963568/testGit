import numpy as np
import matplotlib.pyplot as plt

##制造1000个随机数据
x = np.linspace(60,260,1000)
cigema = 20
miu = 172
#绘制σ=20,μ=172的正态分布
fx = 1 / (cigema * (2 * np.pi)**0.5) * np.exp(-(x - miu)**2 / (2 * cigema**2))
plt.plot(x,fx,color='dodgerblue')
#取μ ~ 1σ的区间
fanwei = x[(x>miu) & (x<miu+cigema)]
#该范围对应的fx
fx2 = 1 / (cigema * (2 * np.pi)**0.5) * np.exp(-(fanwei - miu)**2 / (2 * cigema**2))
#该范围内的曲线与x轴之间的颜色填充
y = np.zeros(fanwei.size)
plt.fill_between(fanwei,fx2,y,fx2 > y,alpha=0.6,color='dodgerblue')
plt.show()
