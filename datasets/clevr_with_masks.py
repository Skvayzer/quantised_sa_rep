from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

class CLEVRwithMasks(Dataset):
    def __init__(self, path_to_dataset):
        self.masks = np.load(path_to_dataset + '_masks.npz')
        self.images = np.load(path_to_dataset + '_images.npz')
        self.visibility = np.load(path_to_dataset + '_visibility.npz')
        self.image_size = self.images[0].shape

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        image = torch.from_numpy(image).float() / 255
        return {
            'image': image,
            'mask': mask
        }