import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class MINE_net1D(BaseNet):

    def __init__(self):
        super().__init__()

        #2d: 15
        #20: 50
        rep = 50
        output_dim=2
        #output_dim=15
        self.rep_dim = output_dim
        

        self.fc1 = nn.Linear(1, rep, bias=True)
        #self.relu1 = nn.ReLU()
        self.relu1 = nn.Tanh()
        
        self.fc2 = nn.Linear(rep, rep, bias=True)
        #self.relu2 = nn.ReLU()
        self.relu2 = nn.Tanh()
        
        self.fc3 = nn.Linear(rep, rep, bias=True)
        #self.relu3 = nn.ReLU()
        self.relu3 = nn.Tanh()

        self.fc4 = nn.Linear(rep, rep, bias=True)
        #self.relu4 = nn.ReLU()
        self.relu4 = nn.Tanh()

        self.fc5 = nn.Linear(rep, output_dim, bias=True)
        
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)        
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        return x
