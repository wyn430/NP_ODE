import math
import numpy as np
from tqdm import tqdm_notebook as tqdm

import torch
from torch import Tensor
from torch import nn
from torch.nn  import functional as F 
from torch.autograd import Variable
import torchvision

from Neural_ODE import *


use_cuda = torch.cuda.is_available()


def norm(dim):
    return nn.BatchNorm2d(dim)

def conv3x3(in_feats, out_feats, stride=1):
    return nn.Conv2d(in_feats, out_feats, kernel_size=(1,3), stride=stride, padding=(0,1), bias=False)



def add_time(in_tensor, t):
    
    bs,c,N,w = in_tensor.shape
    return torch.cat((in_tensor, t.expand(bs, 1, N, w)), dim=1)

class ConvODEF(ODEF):
    def __init__(self, dim):
        super(ConvODEF, self).__init__()
        self.deconv = nn.Conv2d(1, dim, kernel_size=(1,3), stride=1, padding=(0,1), bias=False)
        self.conv1 = conv3x3(dim + 1, dim)       
        self.conv2 = conv3x3(dim + 1, dim)

        
    def forward(self, x, t):

        bs, N, w = x.shape
        x = x.reshape(bs,1,N,w)
        x = self.deconv(x)       
        xt = add_time(x, t)
        h = torch.relu(self.conv1(xt))
        ht = add_time(h, t)
        dxdt = torch.relu(self.conv2(ht))
        dxdt = dxdt.mean(dim=1)

        return dxdt