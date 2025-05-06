import torch, torchvision, functools
from torch.utils.data import DataLoader
from mnist import MNIST
from lenet5 import LeNet5

import torch.serialization
torch.serialization.add_safe_globals([LeNet5])

_original_load = torch.load

def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(*args, **kwargs)

torch.load = _patched_load


def test(dataloader, model):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for img, label in dataloader:
            pred = model(img).argmax(1)
            correct += (pred == label).sum().item()
            total   += label.size(0)
    print(f"test accuracy: {correct / total:.4f}")


def main():
    pad = torchvision.transforms.Pad(2, fill=0)
    mnist_test = MNIST(split="test", transform=pad)
    test_loader = DataLoader(mnist_test, batch_size=1, shuffle=False)

    model = torch.load("LeNet1.pth")
    test(test_loader, model)


if __name__ == '__main__':
    main()