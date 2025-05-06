import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pandas as pd, numpy as np

# ----------------- LeNet‑2 -----------------
class LeNet2(nn.Module):
    def __init__(self):
        super().__init__()
        # Spatial Transformer
        self.loc = nn.Sequential(
            nn.Conv2d(1, 8, 7), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 10, 5), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32), nn.ReLU(),
            nn.Linear(32, 6)
        )
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # 主干 CNN
        self.conv1 = nn.Conv2d(1, 6, 5)      # 32→28
        self.conv2 = nn.Conv2d(6, 16, 5)     # 14→10
        self.conv3 = nn.Conv2d(16, 120, 5)   # 5→1
        self.fc1   = nn.Linear(120, 84)
        self.drop  = nn.Dropout(0.25)
        self.fc2   = nn.Linear(84, 10)

    def stn(self, x):
        xs = self.loc(x)
        xs = xs.view(x.size(0), -1)
        theta = self.fc_loc(xs).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)

    def forward(self, x):
        x = self.stn(x)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x)).view(-1, 120)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)

# ------------- transforms -------------
def to_minus1_1(t):          # 顶层函数，避免多进程 pickle 问题
    return t * 2 - 1

train_tf = T.Compose([
    T.Pad(2),
    T.RandomAffine(15, translate=(0.15, 0.15),
                   scale=(0.8, 1.2), shear=10, fill=255),
    T.ToTensor(),
    T.Lambda(to_minus1_1)
])

test_tf = T.Compose([
    T.Pad(2),
    T.ToTensor(),
    T.Lambda(to_minus1_1)
])

train_ds = MNIST('./data_raw', train=True,  download=True, transform=train_tf)
val_ds   = MNIST('./data_raw', train=False, download=True, transform=test_tf)

# Windows 安全起见使用单进程 dataloader
train_ld = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
val_ld   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=0)

# ------------- training utils -------------
def error_rate(loader, model, device):
    wrong = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            wrong += (pred != y).sum().item()
    return wrong / len(loader.dataset)

# ------------- main loop -------------
def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet2().to(dev)

    opt = torch.optim.SGD(model.parameters(), lr=0.01,
                          momentum=0.9, nesterov=True, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
    criterion = nn.CrossEntropyLoss()

    log = []
    for ep in range(1, 21):
        model.train()
        for x, y in train_ld:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
        sched.step()

        tr_err = error_rate(train_ld, model, dev)
        va_err = error_rate(val_ld,   model, dev)
        log.append((ep, tr_err, va_err))
        print(f"Epoch {ep:02d}: train_err {tr_err:.4f}, val_err {va_err:.4f}")

    torch.save(model, "LeNet2.pth")
    print("Saved model -> LeNet2.pth")
    pd.DataFrame(log, columns=["epoch", "train_err", "val_err"]) \
      .to_csv("train2_log.csv", index=False)

if __name__ == "__main__":
    main()