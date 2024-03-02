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

allDf = pd.read_csv('all_results.csv')
allDf['cnn'] = allDf.type == 'CNN'

datasets = ['derma_pt','lidc','lidc_small','derma', \
        'derma_small','derma_smallest','pneumonia','pneumonia_small']

models = allDf.model.unique()
pdb.set_trace()
