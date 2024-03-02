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

r = 4
c = 5

allDf = {}
keys = ['derma_pt','lidc','lidc_small','derma',\
        'derma_small','derma_smallest','pneumonia','pneumonia_small']

models = pd.read_csv('model_names.csv')
models = models.drop_duplicates('model')
types = np.unique(models.type.values)
for t in types:
    print(('%d '+t)%(np.sum(models.type.values == t)))
model_names = np.unique(models.model.values)

for data in keys:
    sns.set_palette("deep")

    print('Processing '+data)
    df = pd.read_csv('model_results_'+data+'.csv')
    df = df[df.memT > 0]
    df = df.drop_duplicates('model')
#    df.loc[:,'num_param'] = np.log(df.loc[:,'num_param'])
    allDf[data] = df
    M = df.shape[0]
    df[list(models.columns)[1:-1]] = models.iloc[:,1:-1]
    print('Found %d models with data'%M)

    plt.clf()
    fig = plt.figure(figsize=(r*5,c*3),constrained_layout=True)
    gs = fig.add_gridspec(r,c)

    ax = fig.add_subplot(gs[0,0])
    plt.title(data)
    plt.subplot(r,c,1)
    sns.scatterplot(data=df,x='num_param',y='train_time',hue='type')
    plt.title('tr.time vs # param.')

    plt.subplot(r,c,2)
    sns.scatterplot(data=df,x='num_param',y='energy',hue='type')
    plt.title('energy vs # param')

    plt.subplot(r,c,3)
    sns.scatterplot(data=df,x='energy',y='train_time',hue='type')
    plt.title('tr.time vs energy')

    plt.subplot(r,c,4)
    sns.scatterplot(data=df,x='energy',y='memR',hue='type')
    plt.title('mem vs energy')

    plt.subplot(r,c,5)
    sns.scatterplot(data=df,x='energy',y='inf_time',hue='type')
    plt.title('Inf. time vs Energy')


    plt.subplot(r,c,6)
    plt.plot(df.num_param,df.test_00,'.',label='Ep.1')
    plt.plot(df.num_param,df.test_04,'.',label='Ep.5')
    plt.plot(df.num_param,df.test_09,'.',label='Ep.10')
    plt.xlabel('# Param.')
    plt.ylabel('Test Perf.')
    plt.legend(loc='lower right')

    plt.subplot(r,c,7)
    plt.plot(df.energy,df.test_00,'.',label='Ep.1')
    plt.plot(df.energy,df.test_04,'.',label='Ep.5')
    plt.plot(df.energy,df.test_09,'.',label='Ep.10')
    plt.xlabel('Tr. Energy')
    plt.legend(loc='lower right')

    plt.subplot(r,c,8)
    plt.plot(df.train_time,df.test_00,'.',label='Ep.1')
    plt.plot(df.train_time,df.test_04,'.',label='Ep.5')
    plt.plot(df.train_time,df.test_09,'.',label='Ep.10')
    plt.xlabel('Tr.time')
    plt.legend(loc='lower right')

    plt.subplot(r,c,9)
    plt.plot(df.memR,df.test_00,'.',label='Ep.1')
    plt.plot(df.memR,df.test_04,'.',label='Ep.5')
    plt.plot(df.memR,df.test_09,'.',label='Ep.10')
    plt.xlabel('GPU Mem.')
    plt.legend(loc='lower right')

    plt.subplot(r,c,10)
    plt.plot(df.inf_time,df.test_00,'.',label='Ep.1')
    plt.plot(df.inf_time,df.test_04,'.',label='Ep.5')
    plt.plot(df.inf_time,df.test_09,'.',label='Ep.10')
    plt.xlabel('Inf. Time')
    plt.legend(loc='lower right')

    sns.set_palette('viridis')
    ax = fig.add_subplot(gs[2,:])
    for i in range(10):
        col = 'test_%02d'%i
        plt.scatter(df.model, df[col],label='Ep.%02d'%(i+1),color=colors[i])#,alpha=0.5+(0.5-0.05*i),s=5*(10-i))
#    if 'derma' in data:
#    plt.ylim([0.19,0.91])

    plt.xticks('')
    plt.legend()


    ax = fig.add_subplot(gs[3,:])
    plt.scatter(df.model, df.num_param)

    plt.xticks(rotation=90,fontsize=10)

    plt.savefig('results_'+data+'.pdf',dpi=300)

torch.save(allDf,'allDf.pt')
### Combined results
plt.clf()
fig = plt.figure(figsize=(r*5,c*3),constrained_layout=True)
gs = fig.add_gridspec(r,c)

ax = fig.add_subplot(gs[0,:])
xAxis = np.arange(M)
plt.title('Difference in performance after Ep.1 with/without pretraining')
plt.plot(xAxis,np.zeros(len(xAxis)),'--',c='grey')
plt.scatter(np.arange(M),allDf['derma_pt']['test_00']-allDf['derma']['test_00'], label='Ep.1',marker='^')
#plt.xticks('')
#plt.ylim([-0.1,0.25])

#ax = fig.add_subplot(gs[1,:])
#plt.title('Difference in performance after Ep.5 with/without pretraining')
#plt.plot(xAxis,np.zeros(len(xAxis)),'--',c='grey')

#plt.scatter(np.arange(M), allDf['derma_pt']['test_04']-allDf['derma']['test_04'], label='Ep.5',marker='^')
#plt.xticks('')
#plt.ylim([-0.1,0.25])

#ax = fig.add_subplot(gs[2,:])
#plt.title('Difference in performance after Ep.10 with/without pretraining')
#plt.plot(xAxis,np.zeros(len(xAxis)),'--',c='grey')
plt.scatter(np.arange(M), allDf['derma_pt']['test_09']-allDf['derma']['test_09'], label='Ep.10',marker='o',color=colors[4])
plt.ylim([-0.1,0.25])
plt.legend()
#plt.xticks('')


ax = fig.add_subplot(gs[3,:])

plt.scatter(allDf['derma'].model, allDf['derma']['test_00'],label='Ep.1',marker='^')
plt.scatter(allDf['derma_pt'].model, allDf['derma_pt']['test_00'],label='Ep.1',color=colors[4])
plt.legend()
plt.xticks(rotation=90,fontsize=10)
plt.savefig('group_results.pdf',dpi=300)



