import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./weights.csv")
plt.plot(data)
# plt.legend()
# plt.xlabel("number")
# plt.ylabel("weights")
plt.show()