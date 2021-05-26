# CSD
This is the official implementation (including [PyTorch version](https://github.com/Booooooooooo/CSD/tree/main/PyTorch%20version) and [MindSpore version](https://github.com/Booooooooooo/CSD/tree/main/MindSpore%20version)) of [Towards Compact Single Image Super-Resolution via Contrastive Self-distillation, IJCAI2021](https://arxiv.org/abs/2105.11683)

## Abstract 

Convolutional neural networks (CNNs) are highly successful for super-resolution (SR) but often require sophisticated architectures with heavy memory cost and computational overhead, significantly restricts their practical deployments on resource-limited devices. In this paper, we proposed a novel contrastive self-distillation (CSD) framework to simultaneously compress and accelerate various off-the-shelf SR models. In particular, a channel-splitting super-resolution network can first be constructed from a target teacher network as a compact student network. Then, we propose a novel contrastive loss to improve the quality of SR images and PSNR/SSIM via explicit knowledge transfer. Extensive experiments demonstrate that the proposed CSD scheme effectively compresses and accelerates several standard SR models such as EDSR, RCAN and CARN.

![model](https://github.com/Booooooooooo/CSD/blob/main/images/model.png)

## Results

![tradeoff](https://github.com/Booooooooooo/CSD/blob/main/images/tradeoff.png)

![table](https://github.com/Booooooooooo/CSD/blob/main/images/table.png)

![visual](https://github.com/Booooooooooo/CSD/blob/main/images/visual.png)

