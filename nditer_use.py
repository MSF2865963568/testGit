import numpy as np

x = np.arange(6).reshape(2, 3)
print(x)
it = np.nditer(x, flags=['f_index'])
print(it.ndim)
# print("type(it):", type(it), "\n it:", it)
# print("*it:", *it)  # 0 1 2 3 4 5
while not it.finished:
    # print("it[0]:", it[0])  # 0,1,2,3,4,5
    # print(it.index, end=",") # 0,2,4,1,3,5,
    print("%d<%s>" % (it[0], it.index))
    it.iternext() # 注：it.iternext()表示进入下一次迭代，如果不加这一句的话，输出的结果就一直都是0 <(0, 0)>且不间断地输出。
