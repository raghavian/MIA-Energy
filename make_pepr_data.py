import pandas as pd
import numpy as np
import pdb

allDf = pd.read_csv('data/all_results.csv')

allDf['cnn'] = allDf.type == 'CNN'
models = pd.read_csv('data/model_names.csv')
models = models.drop_duplicates('model').reset_index()
datasets = ['derma_pt','lidc','lidc_small','derma', \
        'derma_small','derma_smallest','pneumonia','pneumonia_small']

fullDf = allDf[(allDf.dataset == 'derma') | (allDf.dataset == 'derma_pt') |\
        (allDf.dataset=='lidc') | (allDf.dataset=='pneumonia')]
pdb.set_trace()

# PePR-E
Emax = fullDf.energy.max()*1000 # Use the largest training energy across epochs
Emin = fullDf.energy.min()*1000/10 # Use the smallest training energy per epoch

En = (fullDf.energy_conv.values - Emin)/(Emax-Emin)

fullDf.loc[:,'E_n'] = En
fullDf.loc[:,'pepr_e'] = fullDf.best_test.values/(1+En)

# PePR-C
Cmax = fullDf.co2.max() # Use the largest training energy across epochs
Cmin = fullDf.co2.min()/10 # Use the smallest training energy per epoch

Cn = (fullDf.co2_conv.values - Cmin)/(Cmax-Cmin)

fullDf.loc[:,'C_n'] = Cn
fullDf.loc[:,'pepr_c'] = fullDf.best_test.values/(1+Cn)

# PePR-T
Tmax = fullDf.train_time.max() # Use the largest training energy across epochs
Tmin = fullDf.train_time.min()/10 # Use the smallest training energy per epoch

Tn = (fullDf.time_conv.values - Tmin)/(Tmax-Tmin)

fullDf.loc[:,'T_n'] = Tn
fullDf.loc[:,'pepr_t`'] = fullDf.best_test.values/(1+Tn)

fullDf.to_csv('data/full_data_pepr.csv',index=False)

