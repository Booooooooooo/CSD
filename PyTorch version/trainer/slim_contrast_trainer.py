import os
import math
from decimal import Decimal
from glob import glob
import datetime, time
from importlib import import_module
import numpy as np

# import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import utils.utility as utility
from loss.contrast_loss import ContrastLoss, ContrastStyleLoss, ContrastNLoss
from loss.adversarial import Adversarial
from loss.perceptual import PerceptualLoss
from model.edsr import EDSR
from model.rcan import RCAN
from utils.niqe import niqe
from utils.ssim import calc_ssim
from data_aug import GaussianSmoothing, BatchGaussianNoise
from utils.WCT2.transfer import WCT2, transfer, transfer_batch


class SlimContrastiveTrainer:
    def __init__(self, args, loader, device, neg_loader=None):
        self.model_str = args.model.lower()
        self.pic_path = f'./output/{self.model_str}/{args.model_filename}/'
        if not os.path.exists(self.pic_path):
            self.makedirs = os.makedirs(self.pic_path)
        self.teacher_model = args.teacher_model
        self.checkpoint_dir = args.pre_train
        self.model_filename = args.model_filename
        self.model_filepath = f'{self.model_filename}.pth'
        self.writer = SummaryWriter(f'log/{self.model_filename}')

        self.start_epoch = -1
        self.device = device
        self.epochs = args.epochs
        self.init_lr = args.lr
        self.rgb_range = args.rgb_range
        self.scale = args.scale[0]
        self.stu_width_mult = args.stu_width_mult
        self.batch_size = args.batch_size
        self.neg_num = args.neg_num
        self.save_results = args.save_results
        self.self_ensemble = args.self_ensemble
        self.print_every = args.print_every
        self.best_psnr = 0
        self.best_psnr_epoch = -1

        self.loader = loader
        self.neg_loader = neg_loader
        self.wct = WCT2(model_path='./utils/WCT2/model_checkpoints', device=self.device, verbose=True)
        for name, param in self.wct.encoder.named_parameters():
            param.requires_grad=False
        for name, param in self.wct.decoder.named_parameters():
            param.requires_grad = False
        self.style_neg = args.style_neg
        self.mean = [0.404, 0.436, 0.446]
        self.std = [0.288, 0.263, 0.275]

        self.build_model(args)
        self.upsampler = nn.Upsample(scale_factor=self.scale, mode='bicubic')
        self.optimizer = utility.make_optimizer(args, self.model)

        self.t_lambda = args.t_lambda
        self.contra_lambda = args.contra_lambda
        self.ad_lambda = args.ad_lambda
        self.percep_lambda = args.percep_lambda
        self.kd_lambda = args.kd_lambda
        self.mean_outside = args.mean_outside
        self.t_detach = args.contrast_t_detach
        # self.contra_loss = ContrastLoss(args.vgg_weight, args.d_func, self.mean_outside, self.t_detach)
        self.contra_loss = ContrastNLoss(args.vgg_weight, self.device)
        # self.contra_loss = ContrastLoss(args.vgg_weight, args.d_func, is_one=True, t_detach=self.t_detach)
        # self.contra_loss = ContrastStyleLoss(args.vgg_weight, self.device)
        self.l1_loss = nn.L1Loss()
        self.ad_loss = Adversarial(args, 'GAN')
        self.percep_loss = PerceptualLoss()
        # self.kd_loss = lpips.LPIPS(net='vgg')
        self.t_l_remove = args.t_l_remove
        self.gt_as_pos = args.gt_as_pos

        self.blur_sigma = args.blur_sigma
        if self.blur_sigma > 0:
            self.data_aug = transforms.Compose([
                GaussianSmoothing(channels=args.n_colors, kernel_size=3, sigma=args.blur_sigma, device=self.device, p=1),
            ])
        self.noise_sigma = args.noise_sigma
        if self.noise_sigma > 0:
            self.data_aug = transforms.Compose([
                BatchGaussianNoise(self.noise_sigma, p=1.0, device=self.device)
            ])



    def train(self):
        self.model.train()

        total_iter = (self.start_epoch+1)*len(self.loader.loader_train)
        for epoch in range(self.start_epoch + 1, self.epochs):
            if epoch >= self.t_l_remove:
                self.t_lambda = 0

            starttime = datetime.datetime.now()

            lrate = utility.adjust_learning_rate(self.optimizer, epoch, self.epochs, self.init_lr)
            print("[Epoch {}]\tlr:{}\t".format(epoch, lrate))
            psnr, t_psnr = 0.0, 0.0
            step = 0
            for batch, (lr, hr, _,) in enumerate(self.loader.loader_train):
                torch.cuda.empty_cache()
                step += 1
                total_iter += 1
                lr = lr.to(self.device)
                hr = hr.to(self.device)

                self.optimizer.zero_grad()
                teacher_sr = self.model(lr)
                # if self.gt_as_pos:
                #     teacher_sr = hr
                # else:
                #     teacher_sr = self.model(lr)
                student_sr = self.model(lr, self.stu_width_mult)
                l1_loss = self.l1_loss(hr, student_sr)
                teacher_l1_loss = self.l1_loss(hr, teacher_sr)

                ## style transfer
                bic_sample = lr[torch.randperm(self.neg_num), :, :, :]
                bic_sample = self.upsampler(bic_sample)
                contras_loss = 0.0
                if self.style_neg:
                    # bic_sample = torch.flip(lr, [0])
                    # bic_sample = self.upsampler(bic_sample)
                    # contras_loss += self.contra_loss(teacher_sr, student_sr, bic_sample, self.wct)

                    # neg = torch.flip(hr, [0])

                    neg = bic_sample.unsqueeze(1)
                    neg = neg.repeat(1, lr.shape[0], 1, 1, 1)
                    # bic_sample = self.upsampler(lr)
                    # neg = torch.empty(bic_sample.shape[0], self.neg_num, bic_sample.shape[1], bic_sample.shape[2], bic_sample.shape[3])
                    # for i in range(bic_sample.shape[0]):
                    #     if i == 0:
                    #         neg[i] = bic_sample[1:self.neg_num+1, :, :, :]
                    #     elif i >= self.neg_num:
                    #         neg[i] = bic_sample[:self.neg_num, :, :, :]
                    #     else:
                    #         neg[i] = torch.cat((bic_sample[:i, :, :, :], bic_sample[i+1:self.neg_num+1, :, :, :]), 0)
                    # neg = neg.permute(1, 0, 2, 3, 4).to(self.device)
                    neg = transfer_batch(self.wct, self.device, lr, neg).to(self.device)
                    # try:
                        # neg = transfer(self.wct, self.device, hr, neg).to(self.device)
                    # except:
                    #     pass
                        # print("exception when transferring")
                    contras_loss += self.contra_loss(hr, student_sr, neg)

                else:
                    # bic_sample = lr[torch.randperm(self.neg_num), :, :, :]
                    bic_sample = self.upsampler(torch.flip(lr, [0]))
                    # if self.blur_sigma > 0 or self.noise_sigma > 0:
                    #     blur_sample = self.data_aug(lr)
                    #     blur_sample = self.upsampler(blur_sample)

                    ## student baseline as negative
                    blur_sample = None
                    # bic_sample = next(iter(self.neg_loader))
                    # bic_sample = bic_sample.to(self.device)

                    if self.neg_num > 0:
                        if self.gt_as_pos:
                            contras_loss = self.contra_loss(hr, student_sr, bic_sample, blur_sample)
                        else:
                            contras_loss = self.contra_loss(teacher_sr, student_sr, bic_sample, blur_sample)
                # if self.mean_outside:
                #     ## student 和 neg 是一一对应
                #     for _ in range(self.neg_num):
                #         contras_loss += self.contra_loss(teacher_sr, student_sr, neg_sample[torch.randperm(neg_sample.shape[0]), :, :, :]).item()
                #     contras_loss /= self.neg_num
                # else:
                #     contras_loss = self.contra_loss(teacher_sr, student_sr, neg_sample[:self.neg_num])

                loss = l1_loss + self.contra_lambda * contras_loss + self.t_lambda * teacher_l1_loss
                if self.ad_lambda > 0:
                    ad_loss = self.ad_loss(student_sr, hr)
                    loss += self.ad_lambda * ad_loss
                    self.writer.add_scalar('Train/Ad_loss', ad_loss, total_iter)
                if self.percep_lambda > 0:
                    percep_loss = self.percep_loss(hr, student_sr)
                    loss += self.percep_lambda * percep_loss
                    self.writer.add_scalar('Train/Percep_loss', percep_loss, total_iter)
                kd_loss = 0
                if self.kd_lambda > 0:
                    kd_loss = self.percep_loss(student_sr, teacher_sr)
                    loss += self.kd_lambda * kd_loss
                    self.writer.add_scalar('Train/KD_loss', kd_loss, total_iter)
                # print(l1_loss, contras_loss, kd_loss)
                loss.backward()
                self.optimizer.step()

                self.writer.add_scalar('Train/L1_loss', l1_loss, total_iter)
                self.writer.add_scalar('Train/Contras_loss', contras_loss, total_iter)
                self.writer.add_scalar('Train/Teacher_l1_loss', teacher_l1_loss, total_iter)
                self.writer.add_scalar('Train/Total_loss', loss, total_iter)

                student_sr = utility.quantize(student_sr, self.rgb_range)
                psnr += utility.calc_psnr(student_sr, hr, self.scale, self.rgb_range)
                # if not self.gt_as_pos:
                teacher_sr = utility.quantize(teacher_sr, self.rgb_range)
                t_psnr += utility.calc_psnr(teacher_sr, hr, self.scale, self.rgb_range)
                if (batch + 1) % self.print_every == 0:
                    print(
                        f"[Epoch {epoch}/{self.epochs}] [Batch {batch * self.batch_size}/{len(self.loader.loader_train.dataset)}] "
                        f"[psnr {psnr / step}]"
                        f"[t_psnr {t_psnr / step}]"
                    )
                    #neg_sample = utility.quantize(neg_sample, self.rgb_range)
                    utility.save_results(f'result_{batch}', hr, self.scale, width=1, rgb_range=self.rgb_range,
                                         postfix='hr', dir=self.pic_path)
                    utility.save_results(f'result_{batch}', teacher_sr, self.scale, width=1, rgb_range=self.rgb_range,
                                         postfix='t_sr', dir=self.pic_path)
                    utility.save_results(f'result_{batch}', student_sr, self.scale, width=1, rgb_range=self.rgb_range,
                                         postfix='s_sr', dir=self.pic_path)
                    # utility.save_results(f'result_{batch}', neg, self.scale, width=1, rgb_range=self.rgb_range,
                    #                      postfix='neg', dir=self.pic_path)

            print(f"training PSNR @epoch {epoch}: {psnr / step}")

            test_psnr = self.test(self.stu_width_mult)
            if test_psnr > self.best_psnr:
                print(f"saving models @epoch {epoch} with psnr: {test_psnr}")
                self.best_psnr = test_psnr
                self.best_psnr_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_psnr': self.best_psnr,
                    'best_psnr_epoch': self.best_psnr_epoch,
                }, f'{self.checkpoint_dir}{self.model_filepath}')

            endtime = datetime.datetime.now()
            cost = (endtime - starttime).seconds
            print(f"time of epoch{epoch}: {cost}")

    def test(self, width_mult=1):
        self.model.eval()
        with torch.no_grad():
            psnr = 0
            niqe_score = 0
            ssim = 0
            t0 = time.time()

            starttime = datetime.datetime.now()
            for d in self.loader.loader_test:
                # print(d[0], d[1], d[2])
                for lr, hr, filename in d:
                    lr = lr.to(self.device)
                    hr = hr.to(self.device)

                    x = [lr]
                    for tf in 'v', 'h', 't':
                        x.extend([utility.transform(_x, tf, self.device) for _x in x])
                    op = ['', 'v', 'h', 'hv', 't', 'tv', 'th', 'thv']

                    if self.self_ensemble:
                        res = self.model(lr, width_mult)
                        for i in range(1, len(x)):
                            _x = x[i]
                            _sr = self.model(_x, width_mult)
                            for _op in op[i]:
                                _sr = utility.transform(_sr, _op, self.device)
                            res = torch.cat((res, _sr), 0)
                        sr = torch.mean(res, 0).unsqueeze(0)
                    else:
                        sr = self.model(lr, width_mult)

                    sr = utility.quantize(sr, self.rgb_range)
                    if self.save_results:
                        utility.save_results(str(filename), sr, self.scale, width_mult,
                                             self.rgb_range, 'SR',
                                             f'./output/test/{self.model_str}/{self.model_filename}')
                    psnr += utility.calc_psnr(sr, hr, self.scale, self.rgb_range, dataset=d)
                    niqe_score += niqe(sr.squeeze(0).permute(1, 2, 0).cpu().numpy())
                    ssim += calc_ssim(sr, hr, self.scale, dataset=d)

                psnr /= len(d)
                niqe_score /= len(d)
                ssim /= len(d)
                print(width_mult, d.dataset.name, psnr, niqe_score, ssim)

                endtime = datetime.datetime.now()
                cost = (endtime - starttime).seconds
                t1 = time.time()
                total_time = (t1 - t0)
                print(f"time of test: {total_time}")
                return psnr

    def build_model(self, args):
        m = import_module('model.' + self.model_str)
        self.model = getattr(m, self.model_str.upper())(args).to(self.device)
        self.model = nn.DataParallel(self.model, device_ids=range(args.n_GPUs))
        self.load_model()

        # test teacher
        # self.test()

    def load_model(self):
        checkpoint_dir = self.checkpoint_dir
        print(f"[*] Load model from {checkpoint_dir}")
        if not os.path.exists(checkpoint_dir):
            self.makedirs = os.makedirs(checkpoint_dir)

        if not os.listdir(checkpoint_dir):
            print(f"[!] No checkpoint in {checkpoint_dir}")
            return

        model = glob(os.path.join(checkpoint_dir, self.model_filepath))

        no_student = False
        if not model:
            no_student = True
            print(f"[!] No checkpoint ")
            print("Loading pre-trained teacher model")
            model = glob(self.teacher_model)
            if not model:
                print(f"[!] No teacher model ")
                return

        model_state_dict = torch.load(model[0])
        if not no_student:
            self.start_epoch = model_state_dict['epoch']
            self.best_psnr = model_state_dict['best_psnr']
            self.best_psnr_epoch = model_state_dict['best_psnr_epoch']

        self.model.load_state_dict(model_state_dict['model_state_dict'], False)