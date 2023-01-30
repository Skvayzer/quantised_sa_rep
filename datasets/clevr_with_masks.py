import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

class CLEVRwithMasks(Dataset):
    def __init__(self, path_to_dataset, resize):
        data = np.load(path_to_dataset)
        self.masks = torch.squeeze(torch.tensor(data['masks']))
        self.images = torch.tensor(data['images'])
        self.resize = resize
        # self.visibility = data['visibility']
        self.image_size = self.images[0].shape
        self.image_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(resize),
            torchvision.transforms.ToTensor()
        ])
        self.mask_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resize)
        ])
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # print("\n\nATTENTION! item : ", self.images[idx].shape, file=sys.stderr, flush=True)
        # print("\n\nATTENTION! item : ", self.masks[idx].shape, file=sys.stderr, flush=True)
        image = self.image_transform(self.images[idx])
        mask = self.mask_transform(self.masks[idx])

        image = image.float() / 255
        print("\n\nATTENTION! clevr with masks image max/min: ", max(image), min(image), file=sys.stderr, flush=True)
        mask = mask.float() / 255
        return {
            'image': image * 2 - 1,
            'mask': mask
        }