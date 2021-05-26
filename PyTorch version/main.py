import torch
from torch.utils.data import DataLoader

import utils.utility as utility
import data
import loss
from option import args
from trainer.slim_contrast_trainer import SlimContrastiveTrainer
from trainer.seperate_contrast_trainer import SeperateContrastiveTrainer
from trainer.multi_csd_trainer import MultiCSDTrainer
from data.neg_sample import Neg_Dataset
from data.div2k_class import DIV2K_Class

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

from signal import signal, SIGPIPE, SIG_DFL, SIG_IGN
signal(SIGPIPE, SIG_IGN)

if __name__ == '__main__':
    loader = data.Data(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.seperate:
        t = SeperateContrastiveTrainer(args, loader, device)
    else:
        t = SlimContrastiveTrainer(args, loader, device)
    #     t = SlimContrastiveTrainer(args, loader, device, neg_loader)

    if args.model_stat:
        total_param = 0
        if args.seperate:
            for name, param in t.s_model.named_parameters():
                total_param += torch.numel(param)
        else:
            for name, param in t.model.named_parameters():
                # print(name, '      ', torch.numel(param))
                total_param += torch.numel(param)
        print(total_param)



    if not args.test_only:
        t.train()
    else:
        t.sub_test()
        # t.test(args.stu_width_mult)
    checkpoint.done()