# CSD
This is the official implementation (including [PyTorch version](https://github.com/Booooooooooo/CSD/tree/main/PyTorch%20version) and [MindSpore version](https://github.com/Booooooooooo/CSD/tree/main/MindSpore%20version)) of [Towards Compact Single Image Super-Resolution via Contrastive Self-distillation, IJCAI2021](https://arxiv.org/abs/2105.11683)

## Abstract 

Convolutional neural networks (CNNs) are highly successful for super-resolution (SR) but often require sophisticated architectures with heavy memory cost and computational overhead, significantly restricts their practical deployments on resource-limited devices. In this paper, we proposed a novel contrastive self-distillation (CSD) framework to simultaneously compress and accelerate various off-the-shelf SR models. In particular, a channel-splitting super-resolution network can first be constructed from a target teacher network as a compact student network. Then, we propose a novel contrastive loss to improve the quality of SR images and PSNR/SSIM via explicit knowledge transfer. Extensive experiments demonstrate that the proposed CSD scheme effectively compresses and accelerates several standard SR models such as EDSR, RCAN and CARN.

![model](https://github.com/Booooooooooo/CSD/blob/main/images/model.png)

## Results

![tradeoff](https://github.com/Booooooooooo/CSD/blob/main/images/tradeoff.png)

![table](https://github.com/Booooooooooo/CSD/blob/main/images/table.png)

![visual](https://github.com/Booooooooooo/CSD/blob/main/images/visual.png)

## Citation 

If you find the code helpful in you research or work, please cite as:

```@inproceedings{wu2021contrastive,
@misc{wang2021compact,
      title={Towards Compact Single Image Super-Resolution via Contrastive Self-distillation}, 
      author={Yanbo Wang and Shaohui Lin and Yanyun Qu and Haiyan Wu and Zhizhong Zhang and Yuan Xie and Angela Yao},
      year={2021},
      eprint={2105.11683},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements

This code is built on [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch). For the training part of the MindSpore version we referred to [DBPN-MindSpore](https://gitee.com/amythist/DBPN-MindSpore/tree/master), [ModelZoo-RCAN](https://gitee.com/mindspore/models/tree/master/research/cv/RCAN) and the official [tutorial](https://www.mindspore.cn/tutorials/zh-CN/master/index.html). We thank the authors for sharing their codes. 

