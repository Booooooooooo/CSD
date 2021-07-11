import mindspore.ops as ops
from mindspore.nn.loss.loss import _Loss
import mindspore.nn as nn
from losses.contras_loss import ContrastLoss
from option import opt

ops_print = ops.Print()
class Loss(_Loss):
    def __init__(self):
        super(Loss, self).__init__()
        self.l1loss = nn.L1Loss()
        self.contras_loss = ContrastLoss()
        # self.opt = opt
        # ops_print('In Loss', type(self.opt), self.opt.w_loss_vgg7)
        # 不知道为什么这里print出来正确，但下面print self.opt.w_loss_l1就要报错。。。
        self.w_loss_l1 = 1
        self.w_loss_vgg7 = 0.1

    def construct(self, pred, pos, neg, gt):
        l1_loss = self.l1loss(pred, gt)
        contras_loss = self.contras_loss(pred, pos, neg)

        # ops_print(self.opt.w_loss_l1, self.opt.w_loss_vgg7)
        # ops_print('In Loss:', l1_loss, contras_loss)
        # loss = self.opt.w_loss_l1 * l1_loss + self.opt.w_loss_vgg7 * contras_loss
        loss = self.w_loss_l1 * l1_loss + self.w_loss_vgg7 * contras_loss
        return self.get_loss(loss)

class CustomWithLossCell(nn.Cell):
    def __init__(self, net, loss_fn):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._net = net
        self._loss_fn = loss_fn

    def construct(self, data, pos, neg, gt):
        output = self._backbone(data)
        return self._loss_fn(output, pos, neg, gt)

# net = DehazeNet()
# loss = Loss()
# loss_net = CustomWithLossCell(net, loss)