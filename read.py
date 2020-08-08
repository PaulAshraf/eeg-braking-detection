import h5py
import numpy as np
with h5py.File('VPae.mat', 'r') as f:
    for i in range(69):
        word = ''
        for letter in f[f['cnt']['clab'][()][i][0]][()]:
            word += chr(letter[0]) + ''
        print(word)

