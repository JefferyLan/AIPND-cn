# TODO: Build network
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.con1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), # 3*224*224 -> 64*224*224
            nn.MaxPool2d(2),           # 64*224*224 -> 64*112*112
            nn.ReLU()           
        )
        self.con2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1,1), # 64*112*112 -> 128*112*112
            nn.MaxPool2d(2),            # 128*112*112 -> 128*56*56
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 56 * 56, 64), # 128*56*56 -> 64
            nn.ReLU(),
            nn.Linear(64, 48),            # 64 -> 48
            nn.ReLU(),
            nn.Linear(48, 10)             # 48 -> 10（分类种类）
        )

    def forward(self, x):
        out = self.con1(x)
        out = self.con2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def build_network(arch):
    if arch == 'vgg16':
        net = models.vgg16(pretrained=False)
    elif arch == 'vgg13':
        net = models.vgg13(pretrained=False)
    elif arch == 'vgg11':
        net = models.vgg11(pretrained=False)
    elif arch == 'vgg19':
        net = models.vgg19(pretrained=False)
    else:
        net = models.vgg16(pretrained=False)
    #net = MyCNN()   
    
    return net
# define loss function
def loss_funcation(net, lr_input):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = lr_input, momentum = 0.9)
    
    return criterion, optimizer