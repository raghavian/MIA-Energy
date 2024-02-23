import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('model_results.csv')

df = df[df.memT > 0]

M = df.shape[0]
print('Found %d models with data'%M)

plt.figure(figsize=(16,8))

plt.subplot(241)
plt.plot(df.num_param,df.train_time,'x')
plt.title('#param vs tr.time')

plt.subplot(242)
plt.plot(df.num_param,df.energy,'x')
plt.title('#param vs energy')

plt.subplot(243)
plt.plot(df.energy,df.train_time,'x')
plt.title('energy vs tr.time')

plt.subplot(244)
plt.plot(df.energy,df.memR,'x')
plt.title('energy vs mem')

plt.subplot(245)
plt.plot(df.num_param,df.test_00,'.')
plt.plot(df.num_param,df.test_04,'.')
plt.plot(df.num_param,df.test_09,'.')

plt.subplot(246)
plt.plot(df.energy,df.test_00,'.')
plt.plot(df.energy,df.test_04,'.')
plt.plot(df.energy,df.test_09,'.')

plt.subplot(247)
plt.plot(df.train_time,df.test_00,'.')
plt.plot(df.train_time,df.test_04,'.')
plt.plot(df.train_time,df.test_09,'.')

plt.subplot(248)
plt.plot(df.memR,df.test_00,'.')
plt.plot(df.memR,df.test_04,'.')
plt.plot(df.memR,df.test_09,'.')


plt.tight_layout()
plt.savefig('results.pdf',dpi=300)
