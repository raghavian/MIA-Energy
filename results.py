import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from cycler import cycler
import numpy as np
import pdb

params = {'font.size': 14,
#          'font.weight': 'bold',
          'axes.labelsize':14,
          'axes.titlesize':14,
          'axes.labelweight':'bold',
          'axes.titleweight':'bold',
          'legend.fontsize': 14,
          'image.cmap' : 'viridis',
         }
matplotlib.rcParams.update(params)

r = 4
c = 5

for data in ['derma_pt','lidc','derma']:
#    sns.set_palette('deep')
    sns.set_palette("deep")
#    sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True)

    print('Processing '+data)
    df = pd.read_csv('model_results_'+data+'.csv')
    df = df[df.memT > 0]
    M = df.shape[0]
    print('Found %d models with data'%M)

    plt.clf()
    fig = plt.figure(figsize=(r*5,c*3),constrained_layout=True)
    gs = fig.add_gridspec(r,c)

    ax = fig.add_subplot(gs[0,0])
    plt.title(data)
    plt.subplot(r,c,1)
    sns.scatterplot(data=df,x='num_param',y='train_time')
    plt.title('tr.time vs # param.')

    plt.subplot(r,c,2)
    sns.scatterplot(data=df,x='num_param',y='energy')
    plt.title('energy vs # param')

    plt.subplot(r,c,3)
    sns.scatterplot(data=df,x='energy',y='train_time')
    plt.title('tr.time vs energy')

    plt.subplot(r,c,4)
    sns.scatterplot(data=df,x='energy',y='memR')
    plt.title('mem vs energy')

    plt.subplot(r,c,5)
    sns.scatterplot(data=df,x='energy',y='inf_time')
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

#    plt.subplot(r,c,11)
#    plt.plot(lidcDf.test_00,df.test_00,'.',label='Ep.1')
#    plt.plot(lidcDf.test_09,df.test_09,'.',label='Ep.10')

#    plt.xlabel('LIDC')
#    plt.ylabel('Derma')
#    plt.legend(loc='lower right')

#    plt.subplot(r,c,12)
#    plt.plot(lidcDf.test_00,df.test_00,'.',label='Ep.1')
#    plt.plot(lidcDf.test_09,df.test_09,'.',label='Ep.1')

    #plt.ylim([0.4,0.75])
#    plt.xlabel('LIDC')
#    plt.ylabel('Derma')
#    plt.legend(loc='lower right')


    sns.set_palette('viridis')
    ax = fig.add_subplot(gs[2,:])
    for i in range(10):
        col = 'test_%02d'%i
        plt.scatter(df.model, df[col],label='Ep.%02d'%(i+1),alpha=0.5+(0.5-0.05*i),s=5*(10-i))
    if data == 'derma':
        plt.ylim([0.49,0.91])
    #    plt.scatter(df.model, df.test_01,label='Ep.2')
    #    plt.scatter(df.model, df.test_04,label='Ep.3')
    #plt.scatter(df.model, df.test_09,label='Ep.10',marker='s',s=20)

    plt.xticks('')
    plt.legend()
    #plt.xticks(rotation=90)


    ax = fig.add_subplot(gs[3,:])
    plt.scatter(df.model, df.num_param)
    #plt.scatter(df.model, df.memR/df.memR.max(),marker='d')
    #plt.scatter(df.model, df.energy/df.energy.max(),marker='^')

    plt.xticks(rotation=90,fontsize=10)

    #plt.tight_layout()
    plt.savefig('results_'+data+'.pdf',dpi=300)
