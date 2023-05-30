import torch
import torch.nn as nn
import torch.nn.functional as F

class RC2DState(nn.Module):
    def __init__(self):
        super(RC2DState,self).__init__()
        self.l1 = nn.Linear(4,256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 256)
        self.l5 = nn.Linear(256, 5)

    def forward(self, x):
        #x = x.view(-1,2) # Flattern the (n,1,28,28) to (n,2)
        x = F.tanh(self.l1(x))
        x = F.tanh(self.l2(x))
        x = F.tanh(self.l3(x))
        x = F.tanh(self.l4(x))
        return self.l5(x)