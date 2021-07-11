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

from option import opt
from models.config import imagenet_cfg
from models.vgg_model import Vgg

context.set_context(mode=context.GRAPH_MODE,
                    device_target="Ascend",
                    device_id=0)

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
        # model = "./AECRNet_MindSpore/trained_models/vgg19_ImageNet.ckpt"
        model = os.path.join(opt.data_url, 'vgg19_ImageNet.ckpt')
        # model = os.path.join('./trained_models', 'vgg19_ImageNet.ckpt')
        print(model)
        param_dict = load_checkpoint(model)
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

    def construct(self, pred, pos, neg):
        pred_vgg, pos_vgg, neg_vgg = self.vgg(pred), self.vgg(pos), self.vgg(neg)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(pred_vgg)):
            d_ap = self.l1(pred_vgg[i], pos_vgg[i]) ##TODO:实际为pos_vgg[i].detach(), MindSpore不支持detach()
            d_an = self.l1(pred_vgg[i], neg_vgg[i]) ##neg_vgg[i].detach()
            contrastive = d_ap / (d_an + 1e-7)

            loss += self.weights[i] * contrastive
        return self.get_loss(loss)
