import mindspore.dataset.vision.py_transforms as py_vision
from mindspore import Tensor
from mindspore.ops import composite as C
import mindspore.ops as ops
import numpy as np
import mindspore
import mindspore.nn as nn

import sys

def norm(img, vgg=False):
    if vgg:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    transform = py_vision.Normalize(mean, std)
    return transform(img)

def denorm(data, vgg=False):
    if vgg:
        mean = np.array([-2.118, -2.036, -1.804])
        std = np.array([4.367, 4.464, 4.444])
        mean = mean.reshape(3, 1, 1)
        std = std.reshape(3, 1, 1)
        trans = (data - mean) / std
        return Tensor(trans, dtype=mindspore.float32)
    else:
        out = Tensor((data + 1) / 2, dtype=mindspore.float32)
        return C.clip_by_value(out, 0, 1)


def gram_matrix(input):
    a, b, c, d = input.shape  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    matmul = ops.MatMul()
    transpose = ops.Transpose()
    div = ops.Div()
    perm = (1, 0)
    G = matmul(features, transpose(features, perm))  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return div(G, (a * b * c * d))

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

class PSNRMetrics(nn.Metric):
    def __init__(self):
        super(PSNRMetrics, self).__init__()
        self.eps = sys.float_info.min
        self.psnr_net = nn.PSNR()
        self.clear()

    def clear(self):
        self.psnr = 0
        self._samples_num = 0

    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError('PSNR need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))
        # y_pred = self._convert_data(inputs[0])
        # y = self._convert_data(inputs[1])
        y_pred = inputs[0]
        y = inputs[1]
        psnr = self.psnr_net(y_pred, y)
        self.psnr += psnr
        self._samples_num += 1

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return self.psnr / self._samples_num

class SSIMMetrics(nn.Metric):
    def __init__(self):
        super(SSIMMetrics, self).__init__()
        self.eps = sys.float_info.min
        self.ssim_net = nn.SSIM()
        self.clear()

    def clear(self):
        self.ssim = 0
        self._samples_num = 0

    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError('SSIM need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))
        # y_pred = self._convert_data(inputs[0])
        # y = self._convert_data(inputs[1])
        y_pred = inputs[0]
        y = inputs[1]
        ssim = self.ssim_net(y_pred, y)
        self.ssim += ssim
        self._samples_num += 1

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return self.ssim / self._samples_num

metrics = {
    "psnr": PSNRMetrics(),
    "ssim": SSIMMetrics()
}