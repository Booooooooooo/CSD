import sys

from models.common import CustomWithLossCell

from mindspore import nn, Model
from mindspore.train.callback import LossMonitor, ModelCheckpoint, CheckpointConfig

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

def train(train_loader, net, opt):
    print('Start Training...')
    l1_loss = nn.L1Loss()
    net = CustomWithLossCell(net, l1_loss, opt)
    optim = nn.Momentum(net.trainable_params(), learning_rate=opt.lr, momentum=0.9)
    model = Model(net, l1_loss, optim, metrics=metrics)

    config_ckpt = CheckpointConfig(save_checkpoint_steps=16000, keep_checkpoint_max=5)
    ckpt_cb = ModelCheckpoint(prefix='edsr_baseline', directory='output/model', config=config_ckpt)

    model.train(epoch=opt.nEpochs, train_dataset=train_loader, callbacks=[LossMonitor(), ckpt_cb],
                dataset_sink_mode=False)
    # for epoch in range(opt.nEpochs):
    #     model.train(epoch=1, train_dataset=training_data_loader, callbacks=[LossMonitor()], dataset_sink_mode=False)
    #
    #     print(f"[{epoch}]Eval...")
    #     results = model.eval(testing_data_loader, dataset_sink_mode=False)
    #     print(f"PSNR: {results['psnr'].value}, SSIM: {results['ssim'].value}")