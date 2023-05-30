import torch
import torch.nn as nn
import torch.nn.functional as F


class HENState(nn.Module):
    # # larger state prediction network
    # def __init__(self):
    #     super(HENState,self).__init__()
    #     self.l1 = nn.Linear(2,64)
    #     self.l2 = nn.Linear(64, 256)
    #     self.l3 = nn.Linear(256, 64)
    #     self.l4 = nn.Linear(64, 4)
    #     # self.l5 = nn.Linear(120, 10)

    # def __init__(self):
    #     super(HENState,self).__init__()
    #     self.l1 = nn.Linear(2,8)
    #     self.l2 = nn.Linear(8, 16)
    #     self.l3 = nn.Linear(16, 8)
    #     self.l4 = nn.Linear(8, 4)

    def __init__(self):
        super(HENState,self).__init__()
        self.l1 = nn.Linear(2,8)
        self.l2 = nn.Linear(8, 32)
        self.l3 = nn.Linear(32, 8)
        self.l4 = nn.Linear(8, 4)

    def forward(self, x):
        x = x.view(-1,2) # Flattern the (n,1,28,28) to (n,2)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        # x = F.relu(self.l4(x))
        return self.l4(x)
