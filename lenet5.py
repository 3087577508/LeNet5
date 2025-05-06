import torch, torch.nn as nn, torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, use_rbf: bool = False, rbf_path: str = 'rbf_centers.pt'):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, 5)    # 32→28
        self.s2 = nn.AvgPool2d(2)       # 28→14
        self.c3 = nn.Conv2d(6, 16, 5)   # 14→10
        self.s4 = nn.AvgPool2d(2)       # 10→5
        self.c5 = nn.Conv2d(16, 120, 5) # 5→1
        self.f6 = nn.Linear(120, 84)
        self.use_rbf = use_rbf
        if use_rbf:
            self.centers = torch.load(rbf_path)        # [10,84]
            self.j = 0.1
        else:
            self.out = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = self.s2(x)
        x = F.relu(self.c3(x))
        x = self.s4(x)
        x = F.relu(self.c5(x)).view(-1, 120)
        x = F.relu(self.f6(x))            # [B,84]
        if self.use_rbf:
            xx = (x**2).sum(1, keepdim=True)
            cc = (self.centers**2).sum(1)
            xc = x @ self.centers.t()
            return -(xx + cc - 2*xc)
        return self.out(x)