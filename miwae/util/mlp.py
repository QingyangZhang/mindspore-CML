
import mindspore.nn as nn
import mindspore
import mindvision
from torch import nn, optim
from mindvision import transforms
import mindspore.nn.functional as F
#import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Normalize(nn.Module):#same as the normalization of CPM
    def __init__(self):
        super(Normalize, self).__init__()
    def forward(self, x):
        b = x + 4
        for i in range(x.shape[0]):
            a = x[i][:]
            mx = a.max()
            mn = a.min()
            mid = a.mean()
            b[i][:] = (a - mid) / (mx - mn)
        return b

class MLP(nn.Module):
    def __init__(self,num_features = 100, lsd_dim = 100):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(num_features,512)
        self.fc2 = nn.Linear(512,128)
        self.fc3 = nn.Linear(128,lsd_dim)
        self.normalize = Normalize()
        
    def forward(self,din):
        dout = nn.functional.relu(self.fc1(din))
        dout = nn.functional.relu(self.fc2(dout))
        dout = self.fc3(dout)
        dout = self.normalize(dout)
        return dout







