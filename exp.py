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

torch.manual_seed(32)
# Globally load device identifier
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
def evaluate(loader):
    ### Evaluation function for validation/testing
    vl_acc = torch.Tensor([0.]).to(device)
    vl_loss = 0.
    labelsNp = []
    predsNp = []
    model.eval()

    for i, (inputs, labels) in enumerate(loader):
        b = inputs.shape[0]
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Inference
        scores = model(inputs)
        scores = scores.view(labels.shape).type_as(labels)

        preds = torch.sigmoid(scores.clone())
        loss = loss_fun(scores, labels)
        vl_loss += loss
        vl_acc += accuracy(labels,preds)

    # Compute AUC over the full (valid/test) set
    vl_acc = vl_acc.item()/len(loader)
    vl_loss = vl_loss.item()/len(loader)

    return vl_acc, vl_loss

# Get all pretrained models
models = timm.list_models(pretrained=True)

print('Found %d pretrained models'%len(models))
# Get all unique arch.

uniq = [m.split('_')[0] for m in models]
uniq = np.unique(uniq)
M = len(uniq)

print('Found %d unique pretrained models'%len(uniq))

# Load dataset
dataset = LIDCdataset()

x = dataset[0][0]
dim = x.shape[-1]
nCh = x.shape[0]
print('Using %d size of images'%dim)
N = len(dataset)

train_sampler = SubsetRandomSampler(np.arange(int(0.6*N)))
valid_sampler = SubsetRandomSampler(np.arange(int(0.6*N),int(0.8*N)))
test_sampler = SubsetRandomSampler(np.arange(int(0.8*N),N))

# Initiliaze input dimensions
num_train = len(train_sampler)
num_valid = len(valid_sampler)
num_test = len(test_sampler)
print("Num. train = %d, Num. val = %d, Num. test = %d"%(num_train,num_valid,num_test))

batch_size = 32
# Assign script args to vars

# Initialize dataloaders
loader_train = DataLoader(dataset = dataset, drop_last=False,num_workers=0,
        batch_size=batch_size, pin_memory=True,sampler=train_sampler)
loader_valid = DataLoader(dataset = dataset, drop_last=True,num_workers=0,
        batch_size=batch_size, pin_memory=True,sampler=valid_sampler)
loader_test = DataLoader(dataset = dataset, drop_last=True,num_workers=0,
        batch_size=batch_size, pin_memory=True,sampler=test_sampler)

nValid = len(loader_valid)
nTrain = len(loader_train)
nTest = len(loader_test)

# Initialize loss and metrics
loss_fun = torch.nn.BCEWithLogitsLoss()
accuracy = binary_accuracy
num_epochs = 10
lr = 1e-4

model = timm.create_model(models[0], pretrained=True,in_chans=nCh,num_classes=1)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
model.to(device)
nParam = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of parameters:%d"%(nParam))
print(f"Using Adam w/ learning rate = {lr:.1e}")

# Miscellaneous initialization
start_time = time.time()
maxAuc = -1
minLoss = 1e3
convIter = 0

# Training starts here
for epoch in range(num_epochs):
#    tracker.epoch_start()
    running_loss = 0.
    running_acc = 0.
    t = time.time()
    model.train()
    predsNp = []
    labelsNp = []
    bNum = 0
    for i, (inputs, labels) in enumerate(loader_train):


        inputs = inputs.to(device)
        labels = labels.to(device)

        for p in model.parameters():
            p.grad = None
        bNum += 1
        b = inputs.shape[0]

        scores = model(inputs)
        #pdb.set_trace()
        scores = scores.view(labels.shape).type_as(labels)
        loss = loss_fun(scores, labels)

        # Backpropagate and update parameters
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = torch.sigmoid(scores.clone())
            running_acc += (accuracy(labels,preds)).item()
            running_loss += loss.item()

        if (i+1) % 10 == 1:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, nTrain, loss.item()))

    tr_acc = running_acc/nTrain

    if epoch == 1:
        t = torch.cuda.get_device_properties(0).total_memory/1e9
        r = torch.cuda.memory_reserved(0)/1e9
        a = torch.cuda.memory_allocated(0)/1e9
        m = torch.cuda.max_memory_allocated(0)/1e9

    # Evaluate on Validation set
    with torch.no_grad():

        best_ts_acc, best_ts_loss = evaluate(loader=loader_test)
        print('Test Set Loss:%.4f\t Acc:%.4f'%(best_ts_loss, best_ts_acc))
#        with open(logFile,"a") as f:
#            print('Test Set Loss:%.4f\tAcc:%.4f'%(best_ts_loss, best_ts_acc),file=f)
#    writeLog(logFile, epoch, running_loss/bNum, tr_acc,
#            vl_loss, vl_acc, ts_loss, ts_acc,  time.time()-t)


