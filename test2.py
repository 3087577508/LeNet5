import sys, torch, torchvision
from torch.utils.data import DataLoader
import mnist
from lenet5 import LeNet5       # whitelist original model in case
from train2 import LeNet2       # improved model class

# ---- 1. white‑list LeNet classes for safe unpickling ----
import torch.serialization
torch.serialization.add_safe_globals([LeNet2, LeNet5])
# make LeNet2 discoverable as __main__.LeNet2 if checkpoint stored that
sys.modules["__main__"].LeNet2 = LeNet2

# ---- 2. patch torch.load so main() stays unchanged ----
_orig_load = torch.load

def _patched_load(*args, **kw):
    kw.setdefault("weights_only", False)  # allow full pickle
    return _orig_load(*args, **kw)

torch.load = _patched_load

# ---- 3. helper: crop to 32×32 & scale to [‑1,1] ----

def _preprocess(x: torch.Tensor) -> torch.Tensor:
    """x shape 1×H×W float32 in [0,255]. If H=W=36 crop to 32, then →[-1,1]."""
    if x.shape[-1] == 36:         # created by extra Pad(2)
        c = 2                     # (36‑32)//2
        x = x[..., c:-c, c:-c]    # center crop 32×32
    return x / 127.5 - 1.0

# ---- 4. required test() implementation ----

def test(dataloader, model):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for img, label in dataloader:
            img = _preprocess(img)           # ensure match training scale
            pred = model(img).argmax(1)
            correct += (pred == label).sum().item()
            total   += label.size(0)
    print("test accuracy:", correct / total)


def main():

    pad=torchvision.transforms.Pad(2,fill=0,padding_mode='constant')

    mnist_test=mnist.MNIST(split="test",transform=pad)

    test_dataloader= DataLoader(mnist_test,batch_size=1,shuffle=False)

    model = torch.load("LeNet2.pth")

    test(test_dataloader,model)

 

if __name__=="__main__":

    main()
