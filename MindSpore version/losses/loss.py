import mindspore.ops as ops
from mindspore.nn.loss.loss import _Loss
import mindspore.nn as nn
from losses.contras_loss import ContrastLoss
from option import opt

ops_print = ops.Print()

class CSD_Loss(_Loss):
    def __init__(self):
        super(CSD_Loss, self).__init__()
        self.l1loss = nn.L1Loss()
        self.contras_loss = ContrastLoss()
        self.contra_lambda = opt.contra_lambda

    def construct(self, stu, tea, neg, gt):
        # tea_l1_loss = self.l1loss(tea, gt)
        l1_loss = self.l1loss(stu, gt)
        contras_loss = self.contras_loss(stu, tea, neg)

        # ops_print(self.opt.w_loss_l1, self.opt.w_loss_vgg7)
        # ops_print('In Loss:', l1_loss, contras_loss)
        # loss = self.opt.w_loss_l1 * l1_loss + self.opt.w_loss_vgg7 * contras_loss
        loss = l1_loss + self.contra_lambda * contras_loss
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