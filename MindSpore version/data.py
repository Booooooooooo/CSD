from os.path import join
from dataset import Dataset_Eval, Dataset_Train, FileName_Eval
import mindspore.dataset.transforms.py_transforms as py_transforms
import mindspore.dataset.vision.py_transforms as py_vision


def transform():
    return py_transforms.Compose([
        py_vision.ToTensor()
    ])


def get_training_set(data_dir, hr, upscale_factor, patch_size, data_augmentation, data_range, isTrain=True):
    hr_dir = join(data_dir, hr)
    return Dataset_Train(hr_dir,patch_size, upscale_factor, data_augmentation, transform=transform(), data_range=data_range, isTrain=isTrain)


def get_eval_set(lr_dir, upscale_factor, data_range):
    return Dataset_Eval(lr_dir, upscale_factor, transform=transform(), data_range=data_range)


def get_eval_fileName(lr_dir):
    return FileName_Eval(lr_dir)
