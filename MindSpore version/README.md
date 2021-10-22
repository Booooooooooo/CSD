## Dependencies

- Python == 3.7.5

- MindSpore: https://www.mindspore.cn/install

- matplotlib

- imageio

- tensorboardX

- opencv-python 

- scipy

- scikit-image

## Train

### Prepare data

We use DIV2K training set as our training data. 

About how to download data, you could refer to [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch)

### Train baseline model

```bash
# train teacher model
python -u train.py --dir_data LOCATION_OF_DATA --data_test Set5 --test_every 1 --filename edsr_baseline --lr 0.0001 --epochs 5000
```

```bash
# train student model (width=0.25)
python -u train.py --dir_data LOCATION_OF_DATA --data_test Set5 --test_every 1 --filename edsr_baseline025 --lr 0.0001 --epochs 5000 --n_feats 64
```

### Train CSD

VGG pre-trained on ImageNet is used in our contrastive loss. Due to copyright reasons, the pre-trained VGG cannot be shared publicly. 

```bash
python -u csd_train.py --dir_data LOCATION_OF_DATA --data_test Set5 --test_every 1 --filename edsr_csd --lr 0.0001 --epochs 5000 --ckpt_path ckpt/TEACHER_MODEL_NAME.ckpt --contra_lambda 200
```

## Test

```bash
python eval.py --dir_data LOCATION_OF_DATA --test_only --ext "img" --data_test B100 --ckpt_path ckpt/MODEL_NAME.ckpt --task_id 0 --scale 4
```