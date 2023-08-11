import torch
import torch.nn as nn
import torch.nn.functional as F


def feature_transform(input):
    thetaIN=input[:,0:1]
    alphaIN=input[:,1:2]
    tIN = 0.01 * input[:,2:3]
    return torch.concat(
        (thetaIN,alphaIN,tIN),
        axis=1,
    )

class BufferControl(nn.Module):
    def __init__(self):
        super(BufferControl,self).__init__()
        self.l1 = nn.Linear(3,64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 64)
        self.l4 = nn.Linear(64, 64)
        self.l5 = nn.Linear(64, 1)


    def forward(self, x):
        x=feature_transform(x)
        x = F.tanh(self.l1(x))
        x = F.tanh(self.l2(x))
        x = F.tanh(self.l3(x))
        x = F.tanh(self.l4(x))
        return self.l5(x)