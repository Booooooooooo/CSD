import os

import mindspore.ops as ops
from mindspore.nn.loss.loss import _Loss
# import mindspore_hub as mshub
import mindspore
from mindspore import context, Tensor, nn
from mindspore.train.model import Model
from mindspore.common import dtype as mstype
from mindspore.dataset.transforms import py_transforms
from mindspore import load_checkpoint, load_param_into_net
from mindspore.ops.functional import stop_gradient
import mindspore.numpy as np

from src.config import imagenet_cfg
from src.vgg_model import Vgg

# context.set_context(mode=context.GRAPH_MODE,
#                     device_target="Ascend",
#                     device_id=0)
# context.set_context(device_id=0)

cfg = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class Vgg19(nn.Cell):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()

        ##load vgg16
        vgg = Vgg(cfg['19'], phase="test", args=imagenet_cfg)
        # model = os.path.join(opt.data_url, 'vgg19_ImageNet.ckpt')
        model = os.path.join('./', 'vgg19_ImageNet.ckpt')
        print(model)
        param_dict = load_checkpoint(model)
        # print(param_dict)
        load_param_into_net(vgg, param_dict)
        vgg.set_train(False)

        vgg_pretrained_features = vgg.layers
        self.slice1 = nn.SequentialCell()
        self.slice2 = nn.SequentialCell()
        self.slice3 = nn.SequentialCell()
        self.slice4 = nn.SequentialCell()
        self.slice5 = nn.SequentialCell()
        for x in range(2):
            self.slice1.append(vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.append(vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.append(vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.append(vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.append(vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.get_parameters():
                param.requires_grad = False

    def construct(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ContrastLoss(_Loss):
    def __init__(self):
        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def construct(self, teacher, student, neg):
        expand_dims = ops.ExpandDims() # unsqueeze算子
        teacher_vgg, student_vgg, neg_vgg = self.vgg(teacher), self.vgg(student), self.vgg(neg)

        loss = 0
        for i in range(len(teacher_vgg)):
            neg_i = expand_dims(neg_vgg[i], 0) # [8, n_feats, w, h]
            # neg_i = neg_i.repeat(student_vgg[i].shape[0], axis=0)  #TODO:1.3版本才会支持Tensor.repeat
            neg_i = np.repeat(neg_i, student_vgg[i].shape[0], axis=0) # [16, 8, n_feats, w, h]
            neg_i = neg_i.transpose((1, 0, 2, 3, 4)) # [8, 16, n_feats, w, h]

            d_ts = self.l1(stop_gradient(teacher_vgg[i]), student_vgg[i])
            # d_sn = (stop_gradient(neg_i) - student_vgg[i]).abs().sum(axis=0).mean() #TODO:1.3版本才支持Tensor.sum
            d_sn = (stop_gradient(neg_i) - student_vgg[i]).abs() # [8, 16, n_feats, w, h]
            # print(d_sn.shape)
            reduceSum = ops.ReduceSum()
            d_sn = reduceSum(d_sn, 0).mean()
            # print(d_sn)

            contrastive = d_ts / (d_sn + 1e-7)
            loss += self.weights[i] * contrastive

        return self.get_loss(loss)
