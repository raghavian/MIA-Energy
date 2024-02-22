import timm
import torch
from torchvision import transforms, datasets
from torch.utils.data import SubsetRandomSampler,random_split, DataLoader
import numpy as np
import pandas as pd
from datasets import LIDCdataset
from tools import makeLogFile,writeLog,dice,dice_loss,binary_accuracy
import time
import pdb

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(1)
# Globally load device identifier
# Get all pretrained models
models = timm.list_models(pretrained=True)
#pdb.set_trace()
print('Found %d pretrained models'%len(models))
# Get all unique arch.

uniq = [m.split('_')[0] for m in models]
uniq = np.unique(uniq)
M = len(uniq)

print('Found %d unique pretrained models'%len(uniq))

df = pd.DataFrame(index=range(M),columns=['model','num_param'])

for mIdx in range(M):
    m = uniq[mIdx]
    mName = np.array(models)[np.flatnonzero(np.core.defchararray.find(models,m)!=-1)]
    mName = mName[0]
    model = timm.create_model(mName) #, pretrained=True)
    df.loc[mIdx]['model'] = mName
    nParam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    df.loc[mIdx]['num_param'] = nParam

    print('Using ',mName,nParam/1e6)
df.to_csv('model_frame.csv',index=False)
