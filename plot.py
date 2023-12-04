import csv
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.decomposition import PCA
import numpy as np
import math
import pandas as pd
import seaborn as sns
# 打开 CSV 文件并读取数据
x, y, z = [], [], []
columns = [i for i in range(1, 6)]
index = [i for i in range(1, 6)]
df = pd.DataFrame(np.random.randn(5, 5), columns=columns, index=index)
with open('excel/robust_SGD/TMF_ml_1m_global_rd_ips_all.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        x.append(int(row[1]))
        y.append(int(row[2]))
        z.append(float(row[4]))
        df.loc[int(row[1]),int(row[2])]=float(row[4])

print(df)
# distances = [math.sqrt(x ** 2 + y ** 2) for x, y in zip(x, y)]
#创建 3D 网格图
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')
plt.pcolormesh(x, y, z, cmap='viridis')
# pdf = pd.DataFrame({'x': x, 'y': y, 'z': z})
# f, ax = plt.subplots(figsize = (10, 4))
# cmap = sns.cubehelix_palette(start = 1, rot = 3, gamma=0.8, as_cmap = True)

# sns.heatmap(df,  linewidths = 0.5, ax = ax,annot=True,fmt='.4f',xticklabels=True, yticklabels=True)
# sns.heatmap(df, cmap='viridis',annot=True)

plt.show()
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# #
# # x, y = np.meshgrid(x, y)
# print(x,y,z)
# ax.bar3d(x, y, z, 0.5, 0.5, 0.5, alpha=1)
# ax.set_zlim3d(0.683, 0.7)


# plt.show()




