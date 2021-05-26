import os
import imageio
import numpy as np
import PIL
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from data import common

class BSD500(data.Dataset):
    def __init__(self, args, name='BSR', train=True):
        super(BSD500, self).__init__()
        self.args = args
        self.name = name
        self.train = train
        self.scale = args.scale[0]
        self.benchmark = False

        if train:
            self.dir_data = os.path.join(args.dir_data, 'BSR/BSDS500/data/images/train/')
        else:
            self.dir_data = os.path.join(args.dir_data, 'BSR/BSDS500/data/images/test/')
        self.dataset_from_folder = common.DatasetFromFolder(self.dir_data, '.jpg')
        self.cropper = transforms.RandomCrop(size=args.patch_size)
        self.resizer = transforms.Resize(size=args.patch_size // args.scale[0], interpolation=PIL.Image.BICUBIC)
        self.totensor = transforms.ToTensor()
    # @staticmethod
    # def to_tensor(patch):
    #     return torch.Tensor(np.asarray(patch).swapaxes(0, 2)).float() / 255 - 0.5
    #
    # @staticmethod
    # def to_image(tensor):
    #     if type(tensor) == torch.Tensor:
    #         tensor = tensor.numpy()
    #
    #     if len(tensor.shape) == 4:
    #         tensor = tensor.swapaxes(1, 3)
    #     elif len(tensor.shape) == 3:
    #         tensor = tensor.swapaxes(0, 2)
    #     else:
    #         raise Exception("Predictions have shape not in set {3,4}")
    #
    #     tensor = (tensor + 0.5) * 255
    #     tensor[tensor > 255] = 255
    #     tensor[tensor < 0] = 0
    #     return tensor.round().astype(int)

    def __getitem__(self, index):
        img, filename = self.dataset_from_folder[index]

        if self.train:
            hr_patch = self.cropper(img)
        else:
            hr_patch = img

        lr_patch = self.resizer(hr_patch)
        hr_patch = self.totensor(hr_patch)
        lr_patch = self.totensor(lr_patch)
        if (not self.args.no_augment) and self.train:
            lr_patch, hr_patch = self.data_aug(lr_patch, hr_patch)

        hr_patch = hr_patch.mul_(self.args.rgb_range)
        lr_patch = lr_patch.mul_(self.args.rgb_range)

        return lr_patch, hr_patch, filename

    def __len__(self):
        return len(self.dataset_from_folder)

    def data_aug(self, lr, hr, hflip=True, rot=True):
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        if hflip:
            lr = torch.flip(lr, [1])
            hr = torch.flip(hr, [1])
            # lr = lr[:, ::-1, :]
            # hr = hr[:, ::-1, :]
        if vflip:
            lr = torch.flip(lr, [2])
            hr = torch.flip(hr, [2])
            # lr = lr[::-1, :, :]
            # hr = hr[::-1, :, :]
        if rot90:
            lr = lr.permute(0, 2, 1)
            hr = hr.permute(0, 2, 1)

        return lr, hr