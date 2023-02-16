import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

class CLEVRwithMasks(Dataset):
    def __init__(self, path_to_dataset, resize, max_objs=6, get_masks=False):
        data = np.load(path_to_dataset)
        raw_images = torch.tensor(data['images'])

        if get_masks:
            # self.masks = torch.squeeze(torch.tensor(data['masks']))
            raw_masks = torch.tensor(data['masks'])
            self.masks = torch.empty(0, 11, 1, 240, 320)

        self.images = torch.empty(0, 3, 240, 320)

        for i, v in enumerate(torch.tensor(data['visibility'])):
            if sum(v) > max_objs+1:
                continue
            self.images = torch.cat((self.images, raw_images[i]))

            if get_masks:
                self.masks = torch.cat((self.masks, raw_masks[i]))

        self.visibility = torch.tensor(data['visibility'])
        self.resize = resize
        self.get_masks = get_masks
        # self.visibility = data['visibility']
        self.image_size = self.images[0].shape
        self.image_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.CenterCrop((192, 192)),
            torchvision.transforms.Resize(resize),
            torchvision.transforms.ToTensor()
        ])
        self.mask_transform = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop((192, 192)),
            torchvision.transforms.Resize(resize)
        ])
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # print("\n\nATTENTION! item : ", self.images[idx].shape, file=sys.stderr, flush=True)
        # print("\n\nATTENTION! item : ", self.masks[idx].shape, file=sys.stderr, flush=True)
        image = self.image_transform(self.images[idx])
        visibility = self.visibility[idx]
        if self.get_masks:
            mask = self.mask_transform(self.masks[idx])
            mask = mask.float() / 255

        # print("\n\nATTENTION! clevr with masks image max/min: ", torch.max(image), torch.min(image), file=sys.stderr, flush=True)
        # print("\n\nATTENTION! clevr with masks mask max/min: ", torch.max(mask), torch.min(mask), file=sys.stderr, flush=True)

        return {
            'image': image * 2 - 1,
            'mask': mask if self.get_masks else [],
            'visibility': visibility
        }