# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""eval script"""
import os
import time
import numpy as np
import mindspore.dataset as ds
from mindspore import Tensor, context
from mindspore.common import dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.nn as nn
from mindspore.train.model import Model

from src.args import args
import src.rcan_model as rcan
# from src.edsr_model import EDSR
from src.edsr_slim import EDSR
from src.data.srdata import SRData
from src.metrics import calc_psnr, quantize, calc_ssim
from src.data.div2k import DIV2K

# device_id = int(os.getenv('DEVICE_ID', '0'))
# context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=device_id, save_graphs=False)
# context.set_context(max_call_depth=10000)
def eval_net(width_mult=1.0):
    """eval"""
    if args.epochs == 0:
        args.epochs = 1e8
    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False
    if args.data_test[0] == 'DIV2K':
        train_dataset = DIV2K(args, name=args.data_test, train=False, benchmark=False)
    else:
        train_dataset = SRData(args, name=args.data_test, train=False, benchmark=False)
    train_de_dataset = ds.GeneratorDataset(train_dataset, ['LR', 'HR'], shuffle=False)
    train_de_dataset = train_de_dataset.batch(1, drop_remainder=True)
    train_loader = train_de_dataset.create_dict_iterator(output_numpy=True)
    #net_m = rcan.RCAN(args)
    net_m = EDSR(args)
    net_m.set_train(False)
    if args.ckpt_path:
        print(f"Load from {args.ckpt_path}")
        param_dict = load_checkpoint(args.ckpt_path)
        load_param_into_net(net_m, param_dict)

    # opt = nn.Adam(net_m.trainable_params(), learning_rate=0.0001, loss_scale=args.loss_scale)
    # model = Model(net_m, nn.L1Loss(), opt)

    print('load mindspore net successfully.')
    num_imgs = train_de_dataset.get_dataset_size()
    psnrs = np.zeros((num_imgs, 1))
    ssims = np.zeros((num_imgs, 1))
    for batch_idx, imgs in enumerate(train_loader):
        lr = imgs['LR']
        hr = imgs['HR']
        lr = Tensor(lr, mstype.float32)
        pred = net_m(lr, Tensor(width_mult))
        pred_np = pred.asnumpy()
        pred_np = quantize(pred_np, 255)
        psnr = calc_psnr(pred_np, hr, args.scale[0], 255.0)
        pred_np = pred_np.reshape(pred_np.shape[-3:]).transpose(1, 2, 0)
        hr = hr.reshape(hr.shape[-3:]).transpose(1, 2, 0)
        ssim = calc_ssim(pred_np, hr, args.scale[0])
        print("current psnr: ", psnr)
        print("current ssim: ", ssim)
        psnrs[batch_idx, 0] = psnr
        ssims[batch_idx, 0] = ssim
    print('Mean psnr of %s x%s is %.4f' % (args.data_test[0], args.scale[0], psnrs.mean(axis=0)[0]))
    print('Mean ssim of %s x%s is %.4f' % (args.data_test[0], args.scale[0], ssims.mean(axis=0)[0]))

def do_eval(eval_ds, eval_net, width_mult=1.0):
    train_loader = eval_ds.create_dict_iterator(output_numpy=True)
    num_imgs = eval_ds.get_dataset_size()
    psnrs = np.zeros((num_imgs, 1))
    ssims = np.zeros((num_imgs, 1))
    for batch_idx, imgs in enumerate(train_loader):
        lr = imgs['LR']
        hr = imgs['HR']
        lr = Tensor(lr, mstype.float32)
        pred = eval_net(lr, Tensor(width_mult))
        pred_np = pred.asnumpy()
        pred_np = quantize(pred_np, 255)
        psnr = calc_psnr(pred_np, hr, args.scale[0], 255.0)
        pred_np = pred_np.reshape(pred_np.shape[-3:]).transpose(1, 2, 0)
        hr = hr.reshape(hr.shape[-3:]).transpose(1, 2, 0)
        ssim = calc_ssim(pred_np, hr, args.scale[0])
        # print("current psnr: ", psnr)
        # print("current ssim: ", ssim)
        psnrs[batch_idx, 0] = psnr
        ssims[batch_idx, 0] = ssim
    print('Mean psnr of %s x%s is %.4f' % (args.data_test[0], args.scale[0], psnrs.mean(axis=0)[0]))
    print('Mean ssim of %s x%s is %.4f' % (args.data_test[0], args.scale[0], ssims.mean(axis=0)[0]))
    return np.mean(psnrs)

if __name__ == '__main__':
    device_id = int(os.getenv('DEVICE_ID', '0'))
    # context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=device_id, save_graphs=False)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", device_id=device_id, save_graphs=False)
    context.set_context(max_call_depth=10000)
    time_start = time.time()
    print("Start eval function!")
    print("Eval 1.0 Teacher")
    eval_net()
    time_end = time.time()
    print('eval_time: %f' % (time_end - time_start))

    print("Eval Student")
    time_start = time.time()
    eval_net(args.stu_width_mult)
    time_end = time.time()
    print('eval_time: %f' % (time_end - time_start))