import numpy as np

# # define
# # dict = {'a': {1, 2, 3}, 'b': {4, 5, 6}}
# # # save
# # np.save('dict.npy', dict)
# # # load
# # dict_load = np.load('init_ps/time_ps.npy', allow_pickle=True)
# #
# # print("dict =", dict_load.item())
# # print("dict['a'] =", dict_load.item()['a'])
# import matplotlib.pyplot as plt
#
# # 准备数据
# values1 = [10, 15, 20, 25]
# values2 = [30, 35, 40, 45]
# categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
#
# # 创建图表
# fig, ax = plt.subplots()
#
# # 绘制柱状图
# ax.bar(range(len(categories)), values1, label='Values 1')
# ax.bar(range(len(categories))[::2], values2, label='Values 2')
#
# # 添加分类标签
# ax.set_xticks(range(len(categories)))
# ax.set_xticklabels(categories)
#
# # 添加图例
# ax.legend()
#
# # 显示图表
# plt.show()
# dict = {'a': {1, 2, 3}, 'b': {4, 5, 6}}
# # save
# np.save('dict.npy', dict)
# # load
# dict_load = np.load('init_ps/time_ps.npy', allow_pickle=True)
#
# print("dict =", dict_load.item())
# print("dict['a'] =", dict_load.item()['a'])
# import matplotlib.pyplot as plt
#
# # 准备数据
# values1 = [10, 15, 20, 25]
# values2 = [30, 35, 40, 45]
# categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
#
# # 创建图表
# fig, ax = plt.subplots()
#
# # 绘制柱状图
# ax.bar(range(len(categories)), values1, label='Values 1')
# ax.bar(range(len(categories))[::2], values2, label='Values 2')
#
# # 添加分类标签
# ax.set_xticks(range(len(categories)))
# ax.set_xticklabels(categories)
#
# # 添加图例
# ax.legend()
#
# # 显示图表
# plt.show()

from itertools import product

# 给定的三个列表
list1 = [1, 2, 3]
list2 = [4, 5, 6]
list3 = [7, 8, 9]

# 要去除的元素
x = 2
y = 5
z = 8

# 计算笛卡尔积
cartesian_product = list(product(list1, list2, list3))
print(cartesian_product)

# 将笛卡尔积转换为集合
cartesian_set = set(cartesian_product)
original_set = set(zip(list1 , list2 , list3))
print(original_set)
# 去除要去除的元素
result = cartesian_set.difference(original_set)
print(result)
