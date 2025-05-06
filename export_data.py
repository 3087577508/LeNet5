import os
from PIL import Image
from torchvision.datasets import MNIST


def export(split: str):
    """Download & export MNIST images and labels."""
    ds = MNIST(root='./data_raw', train=(split == 'train'), download=True)
    images, labels = ds.data, ds.targets

    out_dir = os.path.join('data', split)
    os.makedirs(out_dir, exist_ok=True)
    label_path = os.path.join('data', f'{split}_label.txt')

    with open(label_path, 'w') as f:
        for idx, (img, lbl) in enumerate(zip(images, labels)):
            Image.fromarray(img.numpy(), mode='L').save(os.path.join(out_dir, f'{idx}.png'))
            f.write(f'{int(lbl)}\n')
    print(f'Exported {len(images)} {split} images.')


if __name__ == '__main__':
    export('train')
    export('test')