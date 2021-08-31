from mindspore.train.callback import ModelCheckpoint, Callback, LossMonitor, TimeMonitor, CheckpointConfig, _InternalCallbackParam, RunContext
import mindspore.nn as nn
from mindspore import ParameterTuple, Tensor
import mindspore.ops as ops
import mindspore.numpy as numpy
from mindspore import load_checkpoint, load_param_into_net, save_checkpoint

from losses.loss import CSD_Loss

class NetWithLossCell(nn.Cell):
    def __init__(self, net, neg_num):
        super(NetWithLossCell, self).__init__()
        self.net = net
        self.neg_num = neg_num
        self.l1_loss = nn.L1Loss()
        self.csd_loss = CSD_Loss()

    def construct(self, lr, hr, stu_width_mult, tea_width_mult):
        sr = self.net(lr, stu_width_mult)
        tea_sr = self.net(lr, tea_width_mult)

        resize = ops.ResizeNearestNeighbor((hr.shape[-2], hr.shape[-1]))
        neg = resize(numpy.flip(lr, 0))

        neg = neg[:self.neg_num, :, :, :]
        loss = self.l1_loss(tea_sr, hr) + self.csd_loss(sr, tea_sr, neg, hr)
        return loss

class TrainOneStepCell(nn.Cell):
    def __init__(self, network, optimizer, neg_num, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        # self.weights = ParameterTuple(network.trainable_params())
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        # self.grad = ops.GradOperation(sens_param=True)
        self.sens = sens

        # self.csd_loss = CSD_Loss()
        # self.l1_loss = nn.L1Loss()
        # self.neg_num = neg_num

    def set_sens(self, value):
        self.sens = value

    def construct(self, lr, hr, width_mult, tea_width_mult):
        weights = self.weights
        loss = self.network(lr, hr, width_mult, tea_width_mult)
        # sr = self.network(lr, width_mult)
        # tea_sr = self.network(lr)

        # resize = ops.ResizeBilinear(hr.shape) #TODO:等1.3版本发布才支持
        # resize = ops.ResizeNearestNeighbor((hr.shape[-2], hr.shape[-1]))
        # neg = resize(numpy.flip(lr, 0))

        # neg = neg[:self.neg_num, :, :, :]
        # loss = self.l1_loss(tea_sr, hr) + self.csd_loss(sr, tea_sr, neg, hr)

        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(lr, hr, width_mult, tea_width_mult, sens)
        return ops.depend(loss, self.optimizer(grads)), loss

def load_model(net, opt):
    print("Loading Teacher")
    param_dict = load_checkpoint(opt.teacher_model)
    # # 将参数加载到网络中
    load_param_into_net(net, param_dict)
    # # print("After loading")
    # # for item in net.get_parameters():
    # #     print(item.asnumpy())
    # #     break

def csd_train(train_loader, net, opt):
    print("[CSD] Start Training...")
    optim = nn.Momentum(net.trainable_params(), learning_rate=opt.lr, momentum=0.9)
    # optim = nn.Adam(net.trainable_params(), learning_rate=opt.lr)
    net_with_loss = NetWithLossCell(net, opt.neg_num)
    train_cell = TrainOneStepCell(net_with_loss, optim, opt.neg_num)
    net.set_train()

    ckpt_config = CheckpointConfig(save_checkpoint_steps=10)
    ckpt_cb = ModelCheckpoint(config=ckpt_config, directory='./output/model/',
                              prefix=opt.model_filename)

    cb_params = _InternalCallbackParam()
    cb_params.train_network = net
    cb_params.cur_step_num = 0
    cb_params.batch_num = opt.batchSize
    cb_params.cur_epoch_num = 0

    run_context = RunContext(cb_params)
    ckpt_cb.begin(run_context)

    for epoch in range(0, opt.nEpochs):
        epoch_loss = 0
        for iteration, batch in enumerate(train_loader.create_dict_iterator(), 1):
            lr = batch["input"]
            hr = batch["target"]

            _, loss = train_cell(lr, hr, Tensor(opt.stu_width_mult), Tensor(1))
            # _, loss = train_cell(lr, hr, opt.stu_width_mult, 1)
            epoch_loss += loss

        print(f"Epoch[{epoch}] loss: {epoch_loss.asnumpy()}")

        if (epoch) % 10 == 0:
            print('===> Saving model...')
            save_checkpoint(net, f'./output/model/{opt.model_filename}.ckpt')
            # cb_params.cur_epoch_num = epoch + 1
            # ckpt_cb.step_end(run_context)


