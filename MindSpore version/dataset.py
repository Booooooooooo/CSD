# @author AmythistHe
# @version 1.0
# @description
# @create 2021/3/22 21:18

import mindspore.dataset
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
from utils.utils import norm
import random


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    # y, _, _ = img.split()
    return img


def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in


def get_patch(img_in, img_tar, img_bic, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))
    img_bic = img_bic.crop((ty, tx, ty + tp, tx + tp))

    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, img_bic, info_patch


def augment(img_in, img_tar, img_bic, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        img_bic = ImageOps.flip(img_bic)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            img_bic = ImageOps.mirror(img_bic)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            img_bic = img_bic.rotate(180)
            info_aug['trans'] = True

    return img_in, img_tar, img_bic, info_aug


class Dataset_Train():
    def __init__(self, image_dir, patch_size, upscale_factor, data_augmentation, data_range, transform=None, isTrain=True):
        super(Dataset_Train, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.image_filenames = sorted(self.image_filenames)
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.isTrain = isTrain
        data_range = [r.split('-') for r in data_range.split('/')]
        if isTrain:
            data_range = data_range[0]
        else:
            if len(data_range) > 1:
                data_range = data_range[1]
            else:
                data_range = [1, len(self.image_filenames)]
        self.begin, self.end = list(map(lambda x: int(x), data_range))
        print(self.begin, self.end, data_range)
        self.image_filenames = self.image_filenames[self.begin - 1:self.end]

    def __getitem__(self, index):
        target = load_img(self.image_filenames[index])
        input = target.resize((int(target.size[0] / self.upscale_factor), int(target.size[1] / self.upscale_factor)),
                              Image.BICUBIC)
        bicubic = rescale_img(input, self.upscale_factor)
        if self.isTrain:
            input, target, bicubic, _ = get_patch(input, target, bicubic, self.patch_size, self.upscale_factor)
        if self.data_augmentation:
            input, target, bicubic, _ = augment(input, target, bicubic)
        # (H, W, C) to (C, H, W)
        # The values in the input arrays are rescaled from [0, 255] to [0.0, 1.0]
        if self.transform:
            input = np.squeeze(self.transform(np.array(input)))
            bicubic = np.squeeze(self.transform(np.array(bicubic)))
            target = np.squeeze(self.transform(np.array(target)))
        # input = np.array(input).transpose(2, 0, 1)
        # target_ori = np.array(target).transpose(2, 0, 1)
        # 标准化
        # input = norm(input, vgg=True)
        # target = norm(target_ori, vgg=True)

        # return input, target, bicubic
        return input, target

    def __len__(self):
        return len(self.image_filenames)


class Dataset_Eval():
    def __init__(self, lr_dir, upscale_factor, data_range, transform=None):
        super(Dataset_Eval, self).__init__()
        self.image_filenames = [join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)]
        self.image_filenames = sorted(self.image_filenames)
        self.upscale_factor = upscale_factor
        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        _, file = os.path.split(self.image_filenames[index])

        bicubic = rescale_img(input, self.upscale_factor)

        if self.transform:
            input = np.squeeze(self.transform(np.array(input)))
            bicubic = np.squeeze(self.transform(np.array(bicubic)))
        # 标准化
        input = norm(input, vgg=True)
        return input, bicubic

    def __len__(self):
        return len(self.image_filenames)


class FileName_Eval():
    def __init__(self, lr_dir):
        super(FileName_Eval, self).__init__()
        self.image_filenames = [join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)]

    def __getitem__(self, index):
        _, file = os.path.split(self.image_filenames[index])
        return file

    def __len__(self):
        return len(self.image_filenames)




















