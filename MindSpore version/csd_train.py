from mindspore.train.callback import ModelCheckpoint, Callback, LossMonitor, TimeMonitor, CheckpointConfig, _InternalCallbackParam, RunContext
import mindspore.nn as nn
from mindspore import ParameterTuple, Tensor
import mindspore.ops as ops
import mindspore.numpy as numpy
from mindspore import load_checkpoint, load_param_into_net, save_checkpoint
from mindspore.common import set_seed
from mindspore import context
from mindspore.context import ParallelMode
import mindspore.dataset as ds

import os
import time

from src.metric import PSNR
from src.args import args
from src.data.div2k import DIV2K
from src.data.srdata import SRData
from src.edsr_slim import EDSR
from src.contras_loss import ContrastLoss
# from eval import do_eval

class NetWithLossCell(nn.Cell):
    def __init__(self, net):
        super(NetWithLossCell, self).__init__()
        self.net = net
        self.l1_loss = nn.L1Loss()

    def construct(self, lr, hr, stu_width_mult, tea_width_mult):
        sr = self.net(lr, stu_width_mult)
        tea_sr = self.net(lr, tea_width_mult)
        loss = self.l1_loss(sr, hr) + self.l1_loss(tea_sr, hr)
        return loss

class NetWithCSDLossCell(nn.Cell):
    def __init__(self, net, contrast_w=0, neg_num=0):
        super(NetWithCSDLossCell, self).__init__()
        self.net = net
        self.neg_num = neg_num
        self.l1_loss = nn.L1Loss()
        self.contrast_loss = ContrastLoss()
        self.contrast_w = contrast_w

    def construct(self, lr, hr, stu_width_mult, tea_width_mult):
        sr = self.net(lr, stu_width_mult)
        tea_sr = self.net(lr, tea_width_mult)
        loss = self.l1_loss(sr, hr) + self.l1_loss(tea_sr, hr)

        resize = nn.ResizeBilinear()
        bic = resize(lr, size=(lr.shape[-2] * 4, lr.shape[-1] * 4))
        neg = numpy.flip(bic, 0)
        neg = neg[:self.neg_num, :, :, :]
        loss += self.contrast_w * self.contrast_loss(tea_sr, sr, neg)
        return loss

class TrainOneStepCell(nn.Cell):
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

    def set_sens(self, value):
        self.sens = value

    def construct(self, lr, hr, width_mult, tea_width_mult):
        weights = self.weights
        loss = self.network(lr, hr, width_mult, tea_width_mult)

        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(lr, hr, width_mult, tea_width_mult, sens)
        self.optimizer(grads)
        return loss

def csd_train(train_loader, net, opt):
    set_seed(1)
    device_id = int(os.getenv('DEVICE_ID', '0'))
    print("[CSD] Start Training...")

    step_size = train_loader.get_dataset_size()
    lr = []
    for i in range(0, opt.epochs):
        cur_lr = opt.lr / (2 ** ((i + 1) // 200))
        lr.extend([cur_lr] * step_size)
    optim = nn.Adam(net.trainable_params(), learning_rate=lr, loss_scale=opt.loss_scale)

    # net_with_loss = NetWithLossCell(net)
    net_with_loss = NetWithCSDLossCell(net, args.contra_lambda, args.neg_num)
    train_cell = TrainOneStepCell(net_with_loss, optim)
    net.set_train()
    eval_net = net

    # time_cb = TimeMonitor(data_size=step_size)
    # loss_cb = LossMonitor()
    # metrics = {
    #     "psnr": PSNR(rgb_range=opt.rgb_range, shave=True),
    # }
    # eval_cb = EvalCallBack(eval_net, eval_ds, args.test_every, step_size / opt.batch_size, metrics=metrics,
    #                        rank_id=rank_id)
    # cb = [time_cb, loss_cb]
    # config_ck = CheckpointConfig(save_checkpoint_steps=opt.ckpt_save_interval * step_size,
    #                              keep_checkpoint_max=opt.ckpt_save_max)
    # ckpt_cb = ModelCheckpoint(prefix=opt.filename, directory=opt.ckpt_save_path, config=config_ck)
    # if device_id == 0:
        # cb += [ckpt_cb]

    for epoch in range(0, opt.epochs):
        epoch_loss = 0
        for iteration, batch in enumerate(train_loader.create_dict_iterator(), 1):
            lr = batch["LR"]
            hr = batch["HR"]

            loss = train_cell(lr, hr, Tensor(opt.stu_width_mult), Tensor(1.0))
            epoch_loss += loss

        print(f"Epoch[{epoch}] loss: {epoch_loss.asnumpy()}")
        # with eval_net.set_train(False):
        #     do_eval(eval_ds, eval_net)

        if (epoch) % 10 == 0:
            print('===> Saving model...')
            save_checkpoint(net, f'./ckpt/{opt.filename}.ckpt')
            # cb_params.cur_epoch_num = epoch + 1
            # ckpt_cb.step_end(run_context)


if __name__ == '__main__':
    time_start = time.time()
    device_id = int(os.getenv('DEVICE_ID', '0'))
    rank_id = int(os.getenv('RANK_ID', '0'))
    device_num = int(os.getenv('RANK_SIZE', '1'))
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", save_graphs=False, device_id=device_id)

    train_dataset = DIV2K(args, name=args.data_train, train=True, benchmark=False)
    train_dataset.set_scale(args.task_id)
    print(len(train_dataset))
    train_de_dataset = ds.GeneratorDataset(train_dataset, ["LR", "HR"], num_shards=device_num,
                                           shard_id=rank_id, shuffle=True)
    train_de_dataset = train_de_dataset.batch(args.batch_size, drop_remainder=True)

    eval_dataset = SRData(args, name=args.data_test, train=False, benchmark=True)
    print(len(eval_dataset))
    eval_ds = ds.GeneratorDataset(eval_dataset, ['LR', 'HR'], shuffle=False)
    eval_ds = eval_ds.batch(1, drop_remainder=True)

    # net_m = RCAN(args)
    net_m = EDSR(args)
    print("Init net weights successfully")

    if args.ckpt_path:
        param_dict = load_checkpoint(args.ckpt_path)
        load_param_into_net(net_m, param_dict)
        print("Load net weight successfully")

    csd_train(train_de_dataset, net_m, args)
    time_end = time.time()
    print('train_time: %f' % (time_end - time_start))