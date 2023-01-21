import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

class CLEVRwithMasks(Dataset):
    def __init__(self, path_to_dataset, transform):
        data = np.load(path_to_dataset)
        self.masks = torch.squeeze(torch.tensor(data['masks']))
        self.images = torch.tensor(data['images'])
        self.transform = transform
        # self.visibility = data['visibility']
        self.image_size = self.images[0].shape

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        print("\n\nATTENTION! item : ", self.images[idx].shape, file=sys.stderr, flush=True)
        print("\n\nATTENTION! item : ", self.masks[idx].shape, file=sys.stderr, flush=True)
        image = self.transform(self.images[idx])
        mask = self.transform(self.masks[idx])

        image = torch.from_numpy(image).float() / 255
        return {
            'image': image,
            'mask': mask
        }