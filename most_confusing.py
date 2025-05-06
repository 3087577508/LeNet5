# --------------------------------------------
# 找出 MNIST 测试集中最“自信”的误分类示例（一类一个）
# --------------------------------------------

import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as T
from lenet5 import LeNet5

# ========= 根据训练时的配置调整这两个量 =========
USE_RBF   = False          # 若训练时用 RBF + Eq9，请改为 True
MODEL_PTH = "LeNet1.pth"   # 训练脚本保存的权重文件
# ===============================================

# ---------- 0. 允许反序列化自定义模型 ----------
import torch.serialization
torch.serialization.add_safe_globals([LeNet5])

# ---------- 1. 载入模型（兼容 state_dict 或完整模型） ----------
obj = torch.load(MODEL_PTH, weights_only=False)
if isinstance(obj, torch.nn.Module):
    model = obj
else:  # 旧版只保存 state_dict
    model = LeNet5(use_rbf=USE_RBF)
    model.load_state_dict(obj)
model.eval()

# ---------- 2. 构造与训练一致的变换 ----------
tfm = T.Compose([
    T.Pad(2),                        # 28×28 → 32×32
    T.ToTensor(),                    # [0,1]
    T.Lambda(lambda t: t * 2 - 1)    # → [-1,1]
])
test_ds = MNIST("./data_raw", train=False, download=False, transform=tfm)
test_ld = DataLoader(test_ds, batch_size=1)

# ---------- 3. 遍历测试集，记录最自信误分类 ----------
best = {d: {"conf": -1, "pred": None, "idx": None} for d in range(10)}

with torch.no_grad():
    for idx, (img, y_true) in enumerate(test_ld):
        probs = F.softmax(model(img), dim=1).squeeze(0)   # [10]
        pred  = probs.argmax().item()
        if pred != y_true.item():                         # 误分类
            conf = probs[pred].item()
            d = y_true.item()
            if conf > best[d]["conf"]:
                best[d] = {"conf": conf, "pred": pred, "idx": idx}

# ---------- 4. 打印结果 ----------
print("Most confident misclassifications:")
for d in range(10):
    info = best[d]
    if info["idx"] is not None:
        print(f"  Digit {d}: test‑idx {info['idx']:>5}, "
              f"mis-as {info['pred']}  (conf {info['conf']:.4f})")
    else:
        print(f"  Digit {d}: no misclassification!")
