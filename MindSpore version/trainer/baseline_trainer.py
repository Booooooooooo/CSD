import sys

from mindspore import nn, Model, Tensor
from mindspore.train.callback import LossMonitor, ModelCheckpoint, CheckpointConfig, SummaryCollector

# from utils.utils import metrics
from utils.callback import EvalCallBack
from utils.lr import get_lr


def train(train_loader, net, opt, test_loader):
    print('Start Training...')

    lr_init = opt.lr
    lr_end = 0.00001
    lr_max = 0.1
    warmup_epochs = 5
    lr = get_lr(lr_init=lr_init, lr_end=lr_end, lr_max=lr_max, warmup_epochs=warmup_epochs,
                total_epochs=opt.nEpochs, steps_per_epoch=50, lr_decay_mode='poly')
    lr = Tensor(lr)

    l1_loss = nn.L1Loss()
    optim = nn.Momentum(net.trainable_params(), learning_rate=lr, momentum=0.9)
    # model = Model(net, l1_loss, optim, metrics=metrics)
    model = Model(net, l1_loss, optim)

    config_ckpt = CheckpointConfig(save_checkpoint_steps=50, keep_checkpoint_max=5)
    ckpt_cb = ModelCheckpoint(prefix=opt.model_filename, directory='./output/model', config=config_ckpt)
    epoch_per_eval = {"epoch":[], "psnr":[], "ssim":[]}
    # eval_cb = EvalCallBack(net, test_loader, 1, epoch_per_eval) ##TODO:会超内存
    summary_collector = SummaryCollector(summary_dir=f'./logs/{opt.model_filename}')

    model.train(epoch=opt.nEpochs, train_dataset=train_loader, callbacks=[LossMonitor(), ckpt_cb, summary_collector],
                dataset_sink_mode=False)

    # model_files = os.listdir('/home/work/user-job-dir/workspace/output/model')
    # for name in model_files:
    #     print(name)
    #     mox.file.rename(f'/home/work/user-job-dir/workspace/output/model/{name}',
    #                     f'obs://test-ddag/output/CSD/output/model/{name}')
    # for epoch in range(opt.nEpochs):
    #     model.train(epoch=1, train_dataset=training_data_loader, callbacks=[LossMonitor()], dataset_sink_mode=False)
    #
    #     print(f"[{epoch}]Eval...")
    #     results = model.eval(testing_data_loader, dataset_sink_mode=False)
    #     print(f"PSNR: {results['psnr'].value}, SSIM: {results['ssim'].value}")