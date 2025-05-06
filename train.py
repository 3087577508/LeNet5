import torch, numpy as np, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import MNIST
from torch import optim, nn
from lenet5 import LeNet5

USE_RBF = False        # True → RBF + Eq9, False → Linear + CE
LR       = 1e-3
EPOCHS   = 20
BATCH    = 64

# ---------- loss function（仅 RBF 模式用） ----------
def eq9_loss(out, tgt, j=0.1):
    # out: [B,10], tgt: [B]
    y_p = out.gather(1, tgt.view(-1, 1)).squeeze(1)      # y_true
    mask = torch.ones_like(out, dtype=torch.bool)
    mask.scatter_(1, tgt.view(-1, 1), False)             # 其他类
    sum_term = torch.exp(-out[mask].view(out.size(0), -1)).sum(1)
    const = torch.exp(torch.tensor(-j, dtype=out.dtype, device=out.device))
    return (y_p + torch.log(const + sum_term)).mean()

# ---------- 训练入口 ----------
def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet5(use_rbf=USE_RBF).to(dev)
    opt   = optim.Adam(model.parameters(), lr=LR)
    loss_fn = eq9_loss if USE_RBF else nn.CrossEntropyLoss()

    tfm = T.Compose([T.Pad(2), T.ToTensor(), T.Lambda(lambda t: t * 2 - 1)])
    tr_ds = MNIST('./data_raw', True,  download=False, transform=tfm)
    te_ds = MNIST('./data_raw', False, download=False, transform=tfm)
    tr_ld = DataLoader(tr_ds, BATCH, shuffle=True)
    te_ld = DataLoader(te_ds, BATCH)

    tr_err_hist, te_err_hist = [], []

    for ep in range(1, EPOCHS + 1):
        # ----- Train -----
        model.train()
        wrong = 0
        for x, y in tr_ld:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            out = model(x)
            wrong += (out.argmax(1) != y).sum().item()   # 错样本数
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
        tr_err = wrong / len(tr_ds)
        tr_err_hist.append(tr_err)

        # ----- Test -----
        model.eval()
        wrong = 0
        with torch.no_grad():
            for x, y in te_ld:
                pred = model(x.to(dev)).argmax(1).cpu()
                wrong += (pred != y).sum().item()
        te_err = wrong / len(te_ds)
        te_err_hist.append(te_err)

        print(f"Epoch {ep:02d}: train error {tr_err:.4f}, test error {te_err:.4f}")

    # 保存与可视化
    torch.save(model, 'LeNet1.pth')
    np.savetxt('err.csv', np.c_[tr_err_hist, te_err_hist], delimiter=',')
    plt.plot(tr_err_hist, label='train'); plt.plot(te_err_hist, label='test')
    plt.xlabel('Epoch'); plt.ylabel('Error rate'); plt.legend(); plt.savefig('error_plot.png')
    print("Model saved as LeNet1.pth, curve saved as error_plot.png")

if __name__ == '__main__':
    main()
