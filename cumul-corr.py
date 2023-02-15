import sys
sys.path.append('./lib')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
#------local modules
from data import snp500,snp500_individual
from visualize import loss_history



si = snp500_individual("snp500")
si.read_csv()
si.prepare_pd()
table = si.get_pd_table()

#-----correlation matrix
f, ax = plt.subplots(figsize=(10, 8))
corr = table.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
f.savefig("corr_matrix.png")

#-----cumulative distribution
loss_history("20230124_184200_")

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
cmap = get_cmap(100)
for e,j in zip(range(0, 10), range(0, 1000, 50)):
    arr = np.load("./npy/20230124_021913_/generated_time_series_%i_%i.npy"%(e,j))
    arr = arr[:, :, 0]
    arr = arr[0,:]
    for i in range(1,11):
        tmp = np.load("./npy/20230124_021913_/generated_time_series_%i_%i.npy"%(e,j))
        tmp = tmp[:, :, 0]
        tmp = tmp[i,:]
        arr = np.append(arr,tmp)
        cs = np.cumsum(arr)
        plt.plot(cs, c=cmap((random.randint(0,99))))

plt.savefig("cdf_mlp_.png")