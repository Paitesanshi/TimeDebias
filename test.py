import numpy as np

# define
dict = {'a': {1, 2, 3}, 'b': {4, 5, 6}}
# save
np.save('dict.npy', dict)
# load
dict_load = np.load('init_ps/time_ps.npy', allow_pickle=True)

print("dict =", dict_load.item())
print("dict['a'] =", dict_load.item()['a'])
