import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('model_results.csv')
lidcDf = pd.read_csv('model_results_lidc.csv')
lidcDf = lidcDf[lidcDf.memT > 0]

df = df[df.memT > 0]
#df = df[:139]
M = df.shape[0]
print('Found %d models with data'%M)

plt.figure(figsize=(20,12))

r = 3
c = 5

plt.subplot(r,c,1)
sns.scatterplot(data=df,x='num_param',y='train_time')
#plt.plot(df.num_param,df.train_time,'x')
plt.title('tr.time vs # param.')

plt.subplot(r,c,2)
sns.scatterplot(data=df,x='num_param',y='energy')
#plt.plot(df.num_param,df.energy,'x')
plt.title('energy vs # param')

plt.subplot(r,c,3)
sns.scatterplot(data=df,x='energy',y='train_time')
#plt.plot(df.energy,df.train_time,'x')
plt.title('tr.time vs energy')

plt.subplot(r,c,4)
sns.scatterplot(data=df,x='energy',y='memR')
#plt.plot(df.energy,df.memR,'x')
plt.title('mem vs energy')

plt.subplot(r,c,5)
sns.scatterplot(data=df,x='energy',y='inf_time')
#plt.plot(df.energy,df.inf_time,'x')
plt.title('Inf. time vs Energy')


plt.subplot(r,c,6)
plt.plot(df.num_param,df.test_00,'.',label='Ep.1')
plt.plot(df.num_param,df.test_04,'.',label='Ep.5')
plt.plot(df.num_param,df.test_09,'.',label='Ep.10')
#plt.ylim([0.4,0.75])
plt.xlabel('# Param.')
plt.ylabel('Test Perf.')
plt.legend(loc='lower right')

plt.subplot(r,c,7)
plt.plot(df.energy,df.test_00,'.',label='Ep.1')
plt.plot(df.energy,df.test_04,'.',label='Ep.5')
plt.plot(df.energy,df.test_09,'.',label='Ep.10')
#plt.ylim([0.4,0.75])
plt.xlabel('Tr. Energy')
plt.legend(loc='lower right')

plt.subplot(r,c,8)
plt.plot(df.train_time,df.test_00,'.',label='Ep.1')
plt.plot(df.train_time,df.test_04,'.',label='Ep.5')
plt.plot(df.train_time,df.test_09,'.',label='Ep.10')
#plt.ylim([0.4,0.75])
plt.xlabel('Tr.time')
plt.legend(loc='lower right')

plt.subplot(r,c,9)
plt.plot(df.memR,df.test_00,'.',label='Ep.1')
plt.plot(df.memR,df.test_04,'.',label='Ep.5')
plt.plot(df.memR,df.test_09,'.',label='Ep.10')
#plt.ylim([0.4,0.75])
plt.xlabel('GPU Mem.')
plt.legend(loc='lower right')

plt.subplot(r,c,10)
plt.plot(df.inf_time,df.test_00,'.',label='Ep.1')
plt.plot(df.inf_time,df.test_04,'.',label='Ep.5')
plt.plot(df.inf_time,df.test_09,'.',label='Ep.10')
#plt.ylim([0.4,0.75])
plt.xlabel('Inf. Time')
plt.legend(loc='lower right')

plt.subplot(r,c,11)
plt.plot(lidcDf.test_00,df.test_00,'.',label='Ep.1')
plt.plot(lidcDf.test_09,df.test_09,'.',label='Ep.10')

#plt.ylim([0.4,0.75])
plt.xlabel('LIDC')
plt.ylabel('Derma')
plt.legend(loc='lower right')

plt.subplot(r,c,12)
plt.plot(lidcDf.test_00,df.test_00,'.',label='Ep.1')
plt.plot(lidcDf.test_09,df.test_09,'.',label='Ep.1')

#plt.ylim([0.4,0.75])
plt.xlabel('LIDC')
plt.ylabel('Derma')
plt.legend(loc='lower right')



plt.tight_layout()
plt.savefig('results.pdf',dpi=300)
