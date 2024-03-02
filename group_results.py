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
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

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

ms = 50
alpha = 1.0
allDf = pd.read_csv('all_results.csv')
allDf['cnn'] = allDf.type == 'CNN'
models = pd.read_csv('model_names.csv')
models = models.drop_duplicates('model').reset_index()
datasets = ['derma_pt','lidc','lidc_small','derma', \
        'derma_small','derma_smallest','pneumonia','pneumonia_small']

#models = allDf.model.unique()
tmp = allDf.groupby('model').mean().reset_index()
perf = tmp.test_09/tmp.test_09.max()
en = tmp.energy/tmp.energy.max()
PeN = perf/(1+en) 

plt.figure(figsize=(12,5.5))
plt.subplot(121)
sns.scatterplot(y=perf,x=np.log10(tmp.num_param),hue=models.type,style=models.efficient,s=ms,alpha=alpha)
plt.grid(axis='y')
#plt.ylim([perf.min()*0.9,1.05])
plt.ylim([0.45,1.05])
plt.xlabel('Number of trainable parameters (log$_{10}$)')
plt.ylabel('Performance')
plt.tight_layout()
#plt.clf()

plt.subplot(122)
sns.scatterplot(y=PeN,x=np.log10(tmp.num_param),hue=models.type,style=models.efficient,s=ms)
plt.ylim([0.45,1.05])
plt.grid(axis='y')
#plt.ylim([PeN.min()*0.9,1.05])
plt.xlabel('Number of trainable parameters (log$_{10}$)')
plt.ylabel('PeN-score')
plt.tight_layout()
plt.savefig('pen_score.pdf',dpi=300)

xAxis = np.log10(models.num_param)

plt.clf()
plt.figure(figsize=(6,5))
sns.scatterplot(y=tmp.energy/tmp.energy.max(),x=np.log10(tmp.num_param),hue=models.type,style=models.efficient,s=ms)
plt.xlabel('Number of trainable parameters (log$_{10}$)')
plt.ylabel('Energy consumption (kWh)')
plt.grid()
plt.tight_layout()
plt.savefig('models.pdf',dpi=300)

plt.clf()
plt.figure(figsize=(6,5))
sns.scatterplot(y=tmp.test_00/tmp.test_00.max()/(1+en),x=np.log10(tmp.num_param),hue=models.type,style=models.efficient,s=ms)
plt.xlabel('Number of trainable parameters (log$_{10}$)')
plt.ylabel('PeN-score')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('pen_score_01.pdf',dpi=300)

plt.clf()
derma_pt = allDf.loc[allDf.dataset=='derma_pt'].reset_index()
derma_npt = allDf.loc[allDf.dataset=='derma'].reset_index()

perf = derma_pt.test_00.values - derma_npt.test_00.values
plt.figure(figsize=(12,5.5))
plt.subplot(121)
sns.scatterplot(y=perf,x=np.log10(derma_npt.num_param),\
        hue=models.type,style=models.efficient,s=ms)
plt.grid(axis='y')
#plt.ylim([perf.min()*0.9,1.05])
plt.ylim([-0.1,0.25])

plt.xlabel('Number of trainable parameters (log$_{10}$)')
plt.ylabel('Performance difference w/without pretraining')
plt.tight_layout()
#plt.clf()
perf = (perf - perf.min())/(perf.max()-perf.min())
PeN = perf/(1+en)
plt.subplot(122)
sns.scatterplot(y=PeN,x=np.log10(tmp.num_param),hue=models.type,style=models.efficient,s=ms)
plt.grid(axis='y')
#plt.ylim([-0.1,0.25])
#plt.ylim([PeN.min()*0.9,1.05])
plt.xlabel('Number of trainable parameters (log$_{10}$)')
plt.ylabel('PeN-score')
plt.tight_layout()

plt.savefig('pretraining.pdf',dpi=300)

plt.clf()
# Create a ScalarMappable to map displacement magnitudes to colors
disp_mag = derma_pt.test_09.values-derma_npt.test_09.values
disp_mag[disp_mag.argmin()] = 0
norm = Normalize(vmin=min(disp_mag), vmax=max(disp_mag))
cmap = plt.cm.viridis  # You can use any colormap you prefer
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

plt.figure(figsize=(10,5))
#sns.scatterplot(y=perf,x=xAxis,hue=models.type,style=models.efficient,s=ms)
#sns.scatterplot(y=PeN,x=xAxis,hue=models.type,style=models.efficient,s=ms)

# Plot the displacement vectors with colors based on displacement magnitudes
for i in range(len(xAxis)):
    plt.arrow(i, 0,
              0, disp_mag[i],
              head_width=0.08, head_length=0.005,linewidth=2,
              color=cmap(norm(disp_mag[i])))

plt.plot(xAxis,np.zeros(len(xAxis)),markersize=0.01)

plt.savefig('pretraining_displacement.pdf',dpi=300)


