from mindspore.train.callback import Callback

from test import test

class EvalCallBack(Callback):
    def __init__(self, model, eval_dataset, eval_per_epoch, epoch_per_eval):
        self.model = model
        self.eval_dataset = eval_dataset
        self.eval_per_epoch = eval_per_epoch
        self.epoch_per_eval = epoch_per_eval

    def epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % self.eval_per_epoch == 0:
            psnr, ssim = test(self.model, self.eval_dataset)
            self.epoch_per_eval["epoch"].append(cur_epoch)
            self.epoch_per_eval["psnr"].append(psnr)
            self.epoch_per_eval["ssim"].append(ssim)
            print(f"PSNR:{psnr}, SSIM:{ssim}")
