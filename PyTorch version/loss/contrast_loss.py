import torch.nn as nn
import torch
from torch.nn import functional as F
import torch.nn.functional as fnn
from torch.autograd import Variable
import numpy as np
from torchvision import models

import math

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ContrastLoss(nn.Module):
    def __init__(self, weights, d_func, t_detach = False, is_one=False):
        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = weights
        self.d_func = d_func
        self.is_one = is_one
        self.t_detach = t_detach

    def forward(self, teacher, student, neg, blur_neg=None):
        teacher_vgg, student_vgg, neg_vgg, = self.vgg(teacher), self.vgg(student), self.vgg(neg)
        blur_neg_vgg = None
        if blur_neg is not None:
            blur_neg_vgg = self.vgg(blur_neg)
        if self.d_func == "L1":
            self.forward_func = self.L1_forward
        elif self.d_func == 'cos':
            self.forward_func = self.cos_forward

        return self.forward_func(teacher_vgg, student_vgg, neg_vgg, blur_neg_vgg)

    def L1_forward(self, teacher, student, neg, blur_neg=None):
        """
        :param teacher: 5*batchsize*color*patchsize*patchsize
        :param student: 5*batchsize*color*patchsize*patchsize
        :param neg: 5*negnum*color*patchsize*patchsize
        :return:
        """
        loss = 0
        for i in range(len(teacher)):
            neg_i = neg[i].unsqueeze(0)
            neg_i = neg_i.repeat(student[i].shape[0], 1, 1, 1, 1)
            neg_i = neg_i.permute(1, 0, 2, 3, 4)### batchsize*negnum*color*patchsize*patchsize
            if blur_neg is not None:
                blur_neg_i = blur_neg[i].unsqueeze(0)
                neg_i = torch.cat((neg_i, blur_neg_i))


            if self.t_detach:
                d_ts = self.l1(teacher[i].detach(), student[i])
            else:
                d_ts = self.l1(teacher[i], student[i])
            d_sn = torch.mean(torch.abs(neg_i.detach() - student[i]).sum(0))

            contrastive = d_ts / (d_sn + 1e-7)
            loss += self.weights[i] * contrastive
        return loss


    def cos_forward(self, teacher, student, neg, blur_neg=None):
        loss = 0
        for i in range(len(teacher)):
            neg_i = neg[i].unsqueeze(0)
            neg_i = neg_i.repeat(student[i].shape[0], 1, 1, 1, 1)
            neg_i = neg_i.permute(1, 0, 2, 3, 4)
            if blur_neg is not None:
                blur_neg_i = blur_neg[i].unsqueeze(0)
                neg_i = torch.cat((neg_i, blur_neg_i))

            if self.t_detach:
                d_ts = torch.cosine_similarity(teacher[i].detach(), student[i], dim=0).mean()
            else:
                d_ts = torch.cosine_similarity(teacher[i], student[i], dim=0).mean()
            d_sn = self.calc_cos_stu_neg(student[i], neg_i.detach())

            contrastive = -torch.log(torch.exp(d_ts)/(torch.exp(d_sn)+1e-7))
            loss += self.weights[i] * contrastive
        return loss

    def calc_cos_stu_neg(self, stu, neg):
        n = stu.shape[0]
        m = neg.shape[0]

        stu = stu.view(n, -1)
        neg = neg.view(m, n, -1)
        # normalize
        stu = F.normalize(stu, p=2, dim=1)
        neg = F.normalize(neg, p=2, dim=2)
        # multiply
        d_sn = torch.mean((stu * neg).sum(0))
        return d_sn
