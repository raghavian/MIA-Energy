import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from cycler import cycler
import numpy as np
import pdb
import matplotlib._color_data as mcd
cmap = matplotlib.cm.get_cmap('viridis_r')
import torch

colors = [cmap(i) for i in np.linspace(0,1,11)]

params = {'font.size': 14,
#          'font.weight': 'bold',
          'axes.labelsize':14,
          'axes.titlesize':14,
          'axes.labelweight':'bold',
          'axes.titleweight':'bold',
          'legend.fontsize': 14,
         }
matplotlib.rcParams.update(params)

allDf = torch.load('allDf.pt')
keys = ['derma_pt','lidc','lidc_small','derma', \
        'derma_small','derma_smallest','pneumonia','pneumonia_small']

i=0
for k in keys:
    df = allDf[k]
    df = df.drop_duplicates('model')
    df['dataset'] = k
    if i == 0:
        cDf = df
        i += 1
    else:
        cDf = pd.concat((cDf,df),axis=0,ignore_index=True)

cDf = cDf.iloc[:,:-1]
cDf.drop(columns='infer_time')
cDf.to_csv('all_results.csv',index=False)
pdb.set_trace()
