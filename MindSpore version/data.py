# @author AmythistHe
# @version 1.0
# @description
# @create 2021/3/22 21:18

from os.path import join
from dataset import Dataset_DBPN_Eval, Dataset_DBPN, FileName_DBPN_Eval
import mindspore.dataset.transforms.py_transforms as py_transforms
import mindspore.dataset.vision.py_transforms as py_vision


def transform():
    return py_transforms.Compose([
        py_vision.ToTensor()
    ])


def get_training_set(data_dir, hr, upscale_factor, patch_size, data_augmentation):
    hr_dir = join(data_dir, hr)
    return Dataset_DBPN(hr_dir,patch_size, upscale_factor, data_augmentation, transform=transform())


def get_eval_set(lr_dir, upscale_factor):
    return Dataset_DBPN_Eval(lr_dir, upscale_factor, transform=transform())


def get_eval_fileName(lr_dir):
    return FileName_DBPN_Eval(lr_dir)
