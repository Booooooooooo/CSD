# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""train"""
import os
import time
from mindspore import context
from mindspore.context import ParallelMode
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init
from mindspore.common import set_seed
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor, Callback
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from src.args import args
from src.data.div2k import DIV2K
from src.data.srdata import SRData
from src.rcan_model import RCAN
# from src.edsr_model import EDSR
from src.edsr_slim import EDSR
from src.metric import PSNR
from eval import do_eval

# def do_eval(eval_network, ds_val, metrics, rank_id, cur_epoch=None):
#     """
#     do eval for psnr and save hr, sr
#     """
#     eval_network.set_train(False)
#     total_step = ds_val.get_dataset_size()
#     setw = len(str(total_step))
#     begin = time.time()
#     step_begin = time.time()
#     # rank_id = get_rank_id()
#     ds_val = ds_val.create_dict_iterator(output_numpy=True)
#     for i, (lr, hr) in enumerate(ds_val):
#         sr = eval_network(lr)
#         _ = [m.update(sr, hr) for m in metrics.values()]
#         result = {k: m.eval(sync=False) for k, m in metrics.items()}
#         result["time"] = time.time() - step_begin
#         step_begin = time.time()
#         print(f"[{i+1:>{setw}}/{total_step:>{setw}}] rank = {rank_id} result = {result}", flush=True)
#     result = {k: m.eval(sync=True) for k, m in metrics.items()}
#     result["time"] = time.time() - begin
#     if cur_epoch is not None:
#         result["epoch"] = cur_epoch
#     if rank_id == 0:
#         print(f"evaluation result = {result}", flush=True)
#     eval_network.set_train(True)
#     return result

class EvalCallBack(Callback):
    """
    eval callback
    """
    def __init__(self, eval_network, ds_val, eval_epoch_frq, epoch_size, metrics, rank_id, result_evaluation=None):
        self.eval_network = eval_network
        self.ds_val = ds_val
        self.eval_epoch_frq = eval_epoch_frq
        self.epoch_size = epoch_size
        self.result_evaluation = result_evaluation
        self.metrics = metrics
        self.best_result = 0
        self.rank_id = rank_id
        self.eval_network.set_train(False)

    def epoch_end(self, run_context):
        """
        do eval in epoch end
        """
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % self.eval_epoch_frq == 0 or cur_epoch == self.epoch_size:
            # result = do_eval(self.eval_network, self.ds_val, self.metrics, self.rank_id, cur_epoch=cur_epoch)
            result = do_eval(self.ds_val, self.eval_network)
            if self.best_result is None or self.best_result < result:
                self.best_result = result
            if self.rank_id == 0:
                print(f"best evaluation result = {self.best_result}", flush=True)
            if isinstance(self.result_evaluation, dict):
                for k, v in result.items():
                    r_list = self.result_evaluation.get(k)
                    if r_list is None:
                        r_list = []
                        self.result_evaluation[k] = r_list
                    r_list.append(v)


def train():
    """train"""
    set_seed(1)
    device_id = int(os.getenv('DEVICE_ID', '0'))
    rank_id = int(os.getenv('RANK_ID', '0'))
    device_num = int(os.getenv('RANK_SIZE', '1'))
    # context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False, device_id=device_id)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", save_graphs=False, device_id=device_id)

    if device_num > 1:
        init()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          device_num=device_num, global_rank=device_id,
                                          gradients_mean=True)
    if args.modelArts_mode:
        import moxing as mox
        local_data_url = '/cache/data'
        mox.file.copy_parallel(src_url=args.data_url, dst_url=local_data_url)

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
        param_dict = load_checkpoint(args.pth_path)
        load_param_into_net(net_m, param_dict)
        print("Load net weight successfully")
    step_size = train_de_dataset.get_dataset_size()
    lr = []
    for i in range(0, args.epochs):
        cur_lr = args.lr / (2 ** ((i + 1) // 200))
        lr.extend([cur_lr] * step_size)
    opt = nn.Adam(net_m.trainable_params(), learning_rate=lr, loss_scale=args.loss_scale)
    loss = nn.L1Loss()
    loss_scale_manager = DynamicLossScaleManager(init_loss_scale=args.init_loss_scale, \
             scale_factor=2, scale_window=1000)

    eval_net = net_m
    model = Model(net_m, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale_manager)

    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    metrics = {
        "psnr": PSNR(rgb_range=args.rgb_range, shave=True),
    }
    eval_cb = EvalCallBack(eval_net, eval_ds, args.test_every, step_size/args.batch_size, metrics=metrics, rank_id=rank_id)
    cb = [time_cb, loss_cb, eval_cb]
    config_ck = CheckpointConfig(save_checkpoint_steps=args.ckpt_save_interval * step_size,
                                 keep_checkpoint_max=args.ckpt_save_max)
    ckpt_cb = ModelCheckpoint(prefix=args.filename, directory=args.ckpt_save_path, config=config_ck)
    if device_id == 0:
        cb += [ckpt_cb]
    model.train(args.epochs, train_de_dataset, callbacks=cb, dataset_sink_mode=True)


if __name__ == "__main__":
    time_start = time.time()
    train()
    time_end = time.time()
    print('train_time: %f' % (time_end - time_start))
