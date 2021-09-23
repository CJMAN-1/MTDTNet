import os
import warnings

import numpy as np
import matplotlib.pyplot as plt

import torch
from PIL import Image
import torchvision
import torchvision.transforms as transforms


class SyntheticDigits(torch.utils.data.Dataset):
    def __init__(self, root, train, transform=None):
        self.train = train
        self.transform = transform
        if train:
            self.image_file = os.path.join(root, 'train.txt')
        else:
            self.image_file = os.path.join(root, 'valid.txt')

        with open(self.image_file, "r") as fp:
            content = fp.readlines()
        self.mapping = list(map(lambda x: (x[0], int(x[1])), [c.strip().split() for c in content]))

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        image, label = self.mapping[idx]
        image = self.transform(Image.open(image).convert('RGB'))
        return image, label


if __name__ == '__main__':
    # IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    composed_transform = transforms.Compose([transforms.ToTensor(), ])
    dst = SyntheticDigits("./synth", train=True, transform=composed_transform)
    trainloader = torch.utils.data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            # img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
        break
