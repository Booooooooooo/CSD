import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os,sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import random
from PIL import Image
import h5py
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from option import args
from data import common

class DIV2K_Class(data.Dataset):
    def __init__(self, img_dir, gt_dir, name=''):
        super(DIV2K_Class, self).__init__()
        self.name = name
        self.args = args
        self.scale = args.scale[0]
        self.benchmark = False

        self.dir_data = os.path.join(img_dir)
        self.gt_data = os.path.join(gt_dir)

        self.img_from_folder = common.DatasetFromFolder(self.dir_data)
        self.gt_from_folder = common.DatasetFromFolder(self.gt_data)
        print(len(self.img_from_folder), len(self.gt_from_folder))
        # self.cropper = tfs.RandomCrop(size=args.patch_size)
        self.totensor = tfs.ToTensor()

    def __getitem__(self, index):
        img, filename = self.img_from_folder[index]
        img = self.totensor(img)
        gt, gt_filename = self.gt_from_folder[index]
        gt = self.totensor(gt)
        # print(filename, gt_filename)
        # neg = neg.permute(2, 1, 0)
        # neg_patch = common.get_patch(
        #     neg,
        #     patch_size=self.args.patch_size,
        #     scale=self.scale,
        #     input_large=True,
        # )[0]
        # neg_patch = neg_patch.permute(2, 0, 1)

        img = img.mul_(self.args.rgb_range)
        gt = gt.mul_(self.args.rgb_range)
        return img, gt, filename

    def __len__(self):
        return len(self.img_from_folder)

class DIV2K_Class_H5(data.Dataset):
    def __init__(self, img_dir, name=''):
        super(DIV2K_Class_H5, self).__init__()
        self.name = name
        self.args = args
        self.scale = args.scale[0]
        self.benchmark = False

        self.dir_data = os.listdir(img_dir)
        self.imgs = [os.path.join(img_dir, img) for img in self.dir_data]

        print(len(self.imgs))
        self.totensor = tfs.ToTensor()

    def __getitem__(self, index):
        filename = self.imgs[index]
        f = h5py.File(self.imgs[index], 'r')
        lr = f['lr']
        gt = f['gt']

        lr = Image.fromarray(np.array(lr))
        gt = Image.fromarray(np.array(gt))
        lr = self.totensor(lr)
        gt = self.totensor(gt)

        lr = lr.mul_(self.args.rgb_range)
        gt = gt.mul_(self.args.rgb_range)

        return lr, gt, filename

    def __len__(self):
        return len(self.imgs)

# if __name__ == '__main__':
    # neg_loader = DataLoader(dataset=Neg_Dataset('/home/ubuntu/data/hdd1/wyb_dataset/DIV2K/0.25student'), batch_size=10, shuffle=True,
    #                         pin_memory=True, num_workers=4)
    # y = next(iter(neg_loader))
    # print(y.shape)