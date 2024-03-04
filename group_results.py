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
#c1 = 'tab:olive'
#c2 = 'tab:green'
ms = 50
alpha = 1.0
allDf = pd.read_csv('data/all_results.csv')
allDf['cnn'] = allDf.type == 'CNN'
models = pd.read_csv('data/model_names.csv')
models = models.drop_duplicates('model').reset_index()
datasets = ['derma_pt','lidc','lidc_small','derma', \
        'derma_small','derma_smallest','pneumonia','pneumonia_small']

#models = allDf.model.unique()
tmp = allDf.groupby('model').mean().reset_index()
perf = tmp.test_09#/tmp.test_09.max()
en = tmp.energy/tmp.energy.max()
PeN = perf/(1+en) 

#sns.set_palette("colorblind")
plt.figure(figsize=(12,5.5))
plt.subplot(121)
sns.scatterplot(y=perf,x=np.log10(tmp.num_param),hue=models.type,style=models.efficient,s=ms,alpha=alpha)
plt.grid(axis='y')
#plt.ylim([perf.min()*0.9,1.05])
plt.ylim([0.3,0.85])
plt.xlabel('Number of trainable parameters (log$_{10}$)')
plt.ylabel('Performance')
plt.tight_layout()
#plt.clf()

plt.subplot(122)
sns.scatterplot(y=PeN,x=np.log10(tmp.num_param),hue=models.type,style=models.efficient,s=ms)
plt.ylim([0.3,0.85])
#plt.ylim([0.45,1.05])
plt.grid(axis='y')
#plt.ylim([PeN.min()*0.9,1.05])
plt.xlabel('Number of trainable parameters (log$_{10}$)')
plt.ylabel('PePR-E score')
plt.tight_layout()
plt.savefig('results/pen_score.pdf',dpi=300)

xAxis = np.log10(models.num_param)

plt.clf()
plt.figure(figsize=(6,5))
sns.scatterplot(y=tmp.test_00/tmp.test_00.max()/(1+en),x=np.log10(tmp.num_param),hue=models.type,style=models.efficient,s=ms)
plt.xlabel('Number of trainable parameters (log$_{10}$)')
plt.ylabel('PePR-E score')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('results/pen_score_01.pdf',dpi=300)

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
plt.ylabel('$\Delta P$ w/without pretraining')
plt.tight_layout()
#plt.clf()
#perf = (perf - perf.min())/(perf.max()-perf.min())
PeN = perf/(1+en)
plt.subplot(122)
sns.scatterplot(y=PeN,x=np.log10(tmp.num_param),hue=models.type,style=models.efficient,s=ms)
plt.grid(axis='y')
#plt.ylim([-0.1,0.25])
#plt.ylim([PeN.min()*0.9,1.05])
plt.xlabel('Number of trainable parameters (log$_{10}$)')
plt.ylabel('PeN-score')
plt.tight_layout()

plt.savefig('results/pretraining.pdf',dpi=300)

plt.clf()
# Create a ScalarMappable to map displacement magnitudes to colors
disp_mag = (derma_pt.test_09.values-derma_npt.test_09.values)/derma_npt.test_09.values * 100
disp_mag[disp_mag.argmin()] = 0

plt.figure(figsize=(10,4))
colorIdx = models.type.values == 'CNN'
colors = np.empty(len(xAxis),dtype=object)
colors[~colorIdx] = 'tab:orange'
colors[colorIdx] = 'tab:blue'
mrkrIdx = models.efficient.values == 1
markers = np.empty(len(xAxis),dtype=object)
markers[~mrkrIdx] = 'o'
markers[mrkrIdx] = 'x'

plt.bar(x=np.arange(len(xAxis)),height=disp_mag, color=colors) 
[plt.plot(i,disp_mag[i],color='black',alpha=0.7,marker=markers[i],markersize=4,linewidth=0.1) for i in range(len(xAxis))]

plt.plot(colorIdx[colorIdx][0],disp_mag[colorIdx][0],label='CNN',linewidth=4)
plt.plot(colorIdx[~colorIdx][0],disp_mag[~colorIdx][0],label='Other',linewidth=4)

plt.plot(mrkrIdx[mrkrIdx][0],disp_mag[mrkrIdx][0],label='Eff.(Y)',linewidth=0.1,marker='x',color='black',markersize=4,alpha=0.7)
plt.plot(mrkrIdx[~mrkrIdx][0],disp_mag[~mrkrIdx][0],label='Eff.(N)',linewidth=0.1,marker='o',color='black',markersize=4,alpha=0.7)

plt.grid(axis='y')

plt.legend()
plt.xlabel('Model indices (sorted in increasing no. of parameters)')
plt.ylabel('$\Delta P$/$P_{0}$%')
plt.tight_layout()
plt.savefig('results/pretraining_displacement.pdf',dpi=300)

plt.clf()
# Create a ScalarMappable to map displacement magnitudes to colors

plt.figure(figsize=(10,4))

disp = allDf.loc[allDf.dataset=='derma','test_09'].values - allDf.loc[allDf.dataset=='derma_small','test_09'].values
plt.bar(x=np.arange(len(xAxis)),height=disp, color=colors) 
[plt.plot(i,disp[i],color='black',alpha=0.7,marker=markers[i],markersize=4,linewidth=0.1) for i in range(len(xAxis))]

#plt.bar(x=np.arange(len(xAxis)),height=allDf.loc[allDf.dataset=='derma_small','test_09'], color=colors) 
plt.plot(colorIdx[colorIdx][0],disp[colorIdx][0],label='CNN',linewidth=4)
plt.plot(colorIdx[~colorIdx][0],disp[~colorIdx][0],label='Other',linewidth=4)

plt.plot(mrkrIdx[mrkrIdx][0],disp[mrkrIdx][0],label='Eff.(Y)',linewidth=0.1,marker='x',color='black',markersize=4,alpha=0.7)
plt.plot(mrkrIdx[~mrkrIdx][0],disp[~mrkrIdx][0],label='Eff.(N)',linewidth=0.1,marker='o',color='black',markersize=4,alpha=0.7)

plt.grid(axis='y')

plt.legend()
plt.xlabel('Model indices (sorted in increasing no. of parameters)')
plt.ylabel('$\Delta P$')
plt.tight_layout()
plt.savefig('results/dataset_displacement.pdf',dpi=300)

### Influence of dataset size; absolute performance
plt.figure(figsize=(10,4))

plt.grid(axis='y')
perf100 = allDf.loc[allDf.dataset=='derma','test_09'].values 
perf10  = allDf.loc[allDf.dataset=='derma_small','test_09'].values

plt.bar(x=np.arange(len(xAxis)),height=perf100, color=colors,linewidth=4,alpha=0.5) 
plt.bar(x=np.arange(len(xAxis)),height=perf10, color=colors,linewidth=4) 

#[plt.plot(i,disp[i],color='tab:grey',marker=markers[i],markersize=4,linewidth=0.1) for i in range(len(xAxis))]

#plt.bar(x=np.arange(len(xAxis)),height=allDf.loc[allDf.dataset=='derma_small','test_09'], color=colors) 
#plt.plot(colorIdx[colorIdx][0],disp[colorIdx][0],label='CNN',linewidth=4)
#plt.plot(colorIdx[~colorIdx][0],disp[~colorIdx][0],label='Other',linewidth=4)

#plt.plot(mrkrIdx[mrkrIdx][0],disp[mrkrIdx][0],label='Eff.(Y)',linewidth=0.1,marker='x',color='tab:grey',markersize=4)
#plt.plot(mrkrIdx[~mrkrIdx][0],disp[~mrkrIdx][0],label='Eff.(N)',linewidth=0.1,marker='o',color='tab:grey',markersize=4)


plt.legend()
plt.xlabel('Model indices (sorted in increasing no. of parameters)')
plt.ylabel('$\Delta P$')
plt.tight_layout()
plt.savefig('results/dataset_size.pdf',dpi=300)

### Correlation between dataset sizes

plt.figure(figsize=(6,5))
colorIdx = models.type.values == 'CNN'
colors = np.empty(len(xAxis),dtype=object)
colors[~colorIdx] = 'tab:orange'
colors[colorIdx] = 'tab:blue'

for d in ['derma','pneumonia','lidc']:
    plt.clf()
    newDf = models
    newDf['small','full'] = 0
    newDf.loc[:,'small'] = allDf.loc[allDf.dataset==d+'_small','test_09'].values 
    newDf.loc[:,'full'] =  allDf.loc[allDf.dataset==d,'test_09'].values
    sns.scatterplot(data=newDf,x='small',y='full',hue='type',style='efficient')

    #plt.legend()
    #plt.xlabel('Model indices (sorted in increasing no. of parameters)')
    #plt.ylabel('$\Delta P$/$P_{0}$%')
    plt.tight_layout()
    plt.savefig('results/dataset_size_'+d+'.pdf',dpi=300)


### Density plot for PeN score
plt.clf()
plt.figure(figsize=(6,10))
plt.subplot(121)
x = np.linspace(0,1,10)
y = np.linspace(0,1,10)

P, E = np.meshgrid(x,y)
PeN = P/(1+E)
plt.contourf(E,P,PeN,levels=10)#,cmap='RdGy')
plt.colorbar(label='PePR-E score');
plt.xlabel('Normalized Energy Unit, $E_n$')
plt.ylabel('Performance metric, P')

### Model space
#plt.clf()
#plt.figure(figsize=(6,5))
plt.subplot(122)
sns.scatterplot(y=tmp.energy/tmp.energy.max(),x=np.log10(tmp.num_param),hue=models.type,style=models.efficient,s=ms)
plt.xlabel('Number of trainable parameters (log$_{10}$)')
plt.ylabel('Energy consumption (kWh)')
plt.grid()
plt.tight_layout()
plt.savefig('results/models_pepr.pdf',dpi=300)


### Influence of epochs
plt.clf()
plt.figure(figsize=(10,4))

sns.set_palette('viridis')
colors = [cmap(i) for i in np.linspace(0,1,11)]
tmp = allDf[allDf.dataset=='derma']

for i in range(10):
    col = 'test_%02d'%i
    plt.scatter(tmp.model, tmp[col],label='Ep.%02d'%(i+1),color=colors[i],s=15)#,alpha=0.5+(0.5-0.05*i),s=5*(10-i))

#    if 'derma' in data:
#    plt.ylim([0.19,0.91])
plt.xticks('')
ax = plt.subplot(111)
ax.spines[['right','top']].set_visible(False)
plt.xticks(np.arange(0,131,20),np.arange(0,131,20))
plt.legend(bbox_to_anchor=(.975,1.0))
plt.xlabel('Model indices (sorted in increasing no. of parameters)')
plt.ylabel('$P$')
plt.tight_layout()
plt.savefig('results/fine_tuning.pdf',dpi=300)
