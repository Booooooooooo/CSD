from mindspore import nn, Model, ops

import os
from os.path import join

from option import opt
from models.edsr import EDSR
from data import get_training_set

import mindspore.dataset as ds
from mindspore import load_checkpoint, load_param_into_net
from mindspore import context
from mindspore import Tensor

def test(net, test_loader, width_mult=1):
    ## TODO:验证一下这部分有没有梯度，为什么会造成测试结果不一样的问题
    # l1_loss = nn.L1Loss()
    model = Model(net)
    # result = model.eval(test_loader)

    _psnr = nn.PSNR()
    _ssim = nn.SSIM()

    psnr = 0.0
    ssim = 0.0
    bic_psnr = 0.0
    for item in test_loader.create_dict_iterator():
        sr = net(item["input"], Tensor(width_mult))
        # sr = model.predict(item["input"])
        hr = item["target"]

        crop_w = hr.shape[-2] - sr.shape[-2]
        crop_h = hr.shape[-1] - sr.shape[-1]
        if crop_w > 0:
            hr = hr[:, :, int(crop_w/2):-(crop_w - int(crop_w/2)), :]
        if crop_h > 0:
            hr = hr[:, :, :, int(crop_h/2):-(crop_h - int(crop_h/2))]

        psnr += _psnr(sr, hr)
        ssim += _ssim(sr, hr)

        # bilinear = rescale_img(lr, 4)
        # bic_psnr += _psnr(bilinear, hr)

    # print(f"PSNR:{psnr / test_loader.get_dataset_size()}, SSIM:{ssim / test_loader.get_dataset_size()}")
    psnr /= test_loader.get_dataset_size()
    ssim /= test_loader.get_dataset_size()
    return psnr.asnumpy(), ssim.asnumpy()


if __name__ == '__main__':
    context.set_auto_parallel_context(device_num=opt.gpus)
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

    # load
    if not os.listdir(opt.pretrained_path):
        print(f"[!] No checkpoint in {opt.pretrained_path}")
    else:
        model = os.path.join(opt.pretrained_path, opt.model_filename)
        if not model:
            print(f"[!] No checkpoint ")

    print(f"Testing not loaded model")
    print(test(net, testing_data_loader))
    ## Test
    param_dict = load_checkpoint(model)
    # 将参数加载到网络中
    load_param_into_net(net, param_dict)
    # print("After loading")
    # for item in net.get_parameters():
    #     print(item.asnumpy())
    #     break
    print(f"Testing trained model {opt.model_filename}")
    print(test(net, testing_data_loader, opt.stu_width_mult))
    print(test(net, testing_data_loader))




