import torch
from torch.utils.data import DataLoader

import utils.utility as utility
import data
import loss
from option import args
from trainer.slim_contrast_trainer import SlimContrastiveTrainer
from trainer.seperate_contrast_trainer import SeperateContrastiveTrainer
from data.neg_sample import Neg_Dataset
from data.div2k_class import DIV2K_Class

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

from signal import signal, SIGPIPE, SIG_DFL, SIG_IGN
signal(SIGPIPE, SIG_IGN)

if __name__ == '__main__':
    loader = data.Data(args)
    # neg_loader = DataLoader(dataset=Neg_Dataset(args.neg_dir),
    #                         batch_size=args.neg_num, shuffle=True, pin_memory=True, num_workers=args.n_threads,)
    # div2k_loader = DataLoader(dataset=DIV2K_Class('/home/wyb/DIV2K_valid_HR_sub_psnr_LR_class1/', '/home/wyb/DIV2K_valid_HR_sub_psnr_GT_class1/'),
    #                           batch_size=1, shuffle=False)
    # loader.loader_test = [div2k_loader]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.seperate:
        t = SeperateContrastiveTrainer(args, loader, device)
    else:
        t = SlimContrastiveTrainer(args, loader, device)
        # t = SlimContrastiveTrainer(args, loader, device, neg_loader)

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
        t.test(args.stu_width_mult)
    checkpoint.done()