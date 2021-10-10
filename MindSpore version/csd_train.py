from __future__ import print_function

# import moxing as mox
# mox.file.shift('os', 'mox')

import os
from os.path import join
from glob import glob
import numpy as np

from option import opt
from models.edsr import EDSR
from data import get_training_set
from trainer.baseline_trainer import train
from trainer.slim_contrast_trainer import csd_train, load_model
from test import test

import mindspore.dataset as ds
from mindspore import load_checkpoint, load_param_into_net
from mindspore import Parameter, Tensor
from mindspore import context
from mindspore.communication.management import init

if __name__ == '__main__':
    # context.set_auto_parallel_context(device_num=opt.gpus)
    # #distributed training
    # context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    # init("nccl")

    # 创建数据集
    print('===> Loading datasets')
    train_set = get_training_set(join(opt.data_dir, opt.data_train), opt.hr_train_dataset, opt.upscale_factor, opt.patch_size,
                                 opt.data_augmentation, opt.data_range)
    # training_data_loader = ds.GeneratorDataset(source=train_set, column_names=["input", "target", "bicubic"],
    #                                            num_parallel_workers=opt.threads, shuffle=True)
    training_data_loader = ds.GeneratorDataset(source=train_set, column_names=["input", "target"],
                                               num_parallel_workers=opt.threads, shuffle=True)
    training_data_loader = training_data_loader.batch(opt.batchSize, drop_remainder=True)

    if opt.data_test == 'DIV2K':
        hr_valid = 'DIV2K_train_HR'
        data_range = opt.data_range
    else:
        opt.data_dir = join(opt.data_dir, 'benchmark')
        data_range= '0'
        hr_valid = 'HR'
    test_set = get_training_set(join(opt.data_dir, opt.data_test), hr_valid, opt.upscale_factor,opt.patch_size,
                                opt.data_augmentation, data_range, isTrain=False)
    # testing_data_loader = ds.GeneratorDataset(source=test_set, column_names=["input", "target", "bicubic"],
    #                                            num_parallel_workers=opt.threads, shuffle=True)
    testing_data_loader = ds.GeneratorDataset(source=test_set, column_names=["input", "target"],
                                              num_parallel_workers=opt.threads, shuffle=True)
    testing_data_loader = testing_data_loader.batch(1, drop_remainder=False)

    print(len(train_set), len(test_set))

    print('===> Building model ', opt.model)
    if opt.model == 'edsr':
        net = EDSR(opt)
        # print("Before loading")
        # dict = net.parameters_dict()
        # print(dict)
        # for item in dict:
        #     print(item, dict[item].asnumpy())
        #     break
    else:
        print('Not implemented...')

    load_model(net, opt)
    print(f"Testing trained model {opt.teacher_model}")
    # print(test(net, testing_data_loader, opt.stu_width_mult))
    # print(test(net, testing_data_loader, Tensor(1)))

    load_checkpoint(net, opt)
    ### Train CSD
    csd_train(training_data_loader, net, opt)


