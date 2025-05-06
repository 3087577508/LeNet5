import os, argparse, numpy as np, torch
from PIL import Image

def main(data_folder: str, output: str):
    img_dir = os.path.join(data_folder, 'train')
    label_file = os.path.join(data_folder, 'train_label.txt')
    if not (os.path.isdir(img_dir) and os.path.isfile(label_file)):
        raise FileNotFoundError('export_data.py 未生成所需文件')

    with open(label_file) as f:
        txt = f.read().strip()
    labels = [int(ch) for ch in txt.split()] if any(c.isspace() for c in txt) else [int(c) for c in txt]

    centers = []
    for d in range(10):
        idxs = [i for i, l in enumerate(labels) if l == d]
        bitmaps = []
        for idx in idxs:
            im = Image.open(os.path.join(img_dir, f'{idx}.png')).convert('L').resize((12, 7), Image.LANCZOS)
            bitmaps.append(np.array(im, np.float32).flatten())
        centers.append(np.mean(bitmaps, axis=0))

    torch.save(torch.tensor(np.stack(centers)), output)
    print('saved', output)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_folder', default='data')
    p.add_argument('--output', default='rbf_centers.pt')
    a = p.parse_args()
    main(a.data_folder, a.output)