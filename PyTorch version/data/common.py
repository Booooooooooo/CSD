import random
import string

import numpy as np
import skimage.color as sc

import os
import imageio
import numpy as np
import PIL

import torch
import torch.utils.data as data

def get_patch(*args, patch_size=96, scale=2, multi=False, input_large=False):
    ih, iw = args[0].shape[:2]

    if not input_large:
        p = scale if multi else 1
        tp = p * patch_size
        ip = tp // scale
    else:
        tp = patch_size
        ip = patch_size

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    if not input_large:
        tx, ty = scale * ix, scale * iy
    else:
        tx, ty = ix, iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]

    return ret

def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]

def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]

def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return [_augment(a) for a in args]

def is_image_file(filename, extension):
    return filename.endswith(extension)

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, extension = ".png"):
        super(DatasetFromFolder, self).__init__()
        self.image_dir = image_dir
        self.image_filenames = [x for x in sorted(os.listdir(image_dir)) if is_image_file(x, extension)]
        self.extension = extension

    def __getitem__(self, index):
        #return imageio.imread(os.path.join(self.image_dir, self.image_filenames[index])), self.image_filenames[index]
        filename = self.image_filenames[index]
        filename = filename.rstrip(self.extension)
        return PIL.Image.fromarray(imageio.imread(os.path.join(self.image_dir, self.image_filenames[index]))), filename

    def __len__(self):
        return len(self.image_filenames)