# confusion_matrix.py
import torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as T
from lenet5 import LeNet5
import torch.serialization

USE_RBF   = False          # 若训练用 RBF 改 True
MODEL_PTH = "LeNet1.pth"

# 1) 允许反序列化
torch.serialization.add_safe_globals([LeNet5])

# 2) 加载模型（兼容 state_dict 或完整模型）
obj = torch.load(MODEL_PTH, weights_only=False)
model = obj if isinstance(obj, torch.nn.Module) else LeNet5(use_rbf=USE_RBF)
if not isinstance(obj, torch.nn.Module):
    model.load_state_dict(obj)
model.eval()

# 3) 准备测试集
tfm = T.Compose([T.Pad(2), T.ToTensor(), T.Lambda(lambda t: t*2-1)])
test_ds = MNIST("./data_raw", train=False, download=False, transform=tfm)
test_ld = DataLoader(test_ds, batch_size=128)

# 4) 计算混淆矩阵
cm = np.zeros((10, 10), dtype=int)
with torch.no_grad():
    for x, y in test_ld:
        preds = model(x).argmax(1).cpu()
        for t, p in zip(y, preds):
            cm[t.item(), p.item()] += 1

# 5) 打印与保存
df = pd.DataFrame(cm,
                  index=[f"True {i}" for i in range(10)],
                  columns=[f"Pred {i}" for i in range(10)])
print(df)
df.to_csv("confusion_matrix.csv")
print("Saved to confusion_matrix.csv")
