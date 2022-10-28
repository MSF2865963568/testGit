import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import torch
file = "test11.txt"
lines = f.readlines()
for line in lines:
    line = line.strip("\n")
    line = line.split(" ")
    line = [float(x) for x in line]

data = np.loadtxt("test11.txt")

# arr4=np.array( ,float)
# a = arr4.flatten()
print(data)

