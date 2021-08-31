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
python train.py --data_dir LOCATION_OF_DATA  --nEpochs 800 --model_filename edsr_baseline --rgb_range 1
```

```bash
# train student model (width=0.25)
python train.py --data_dir LOCATION_OF_DATA  --nEpochs 800 --model_filename edsr_baseline025 --rgb_range 1 --n_feats 64
```

### Train CSD

VGG pre-trained on ImageNet is used in our contrastive loss. Due to copyright reasons, the pre-trained VGG cannot be shared publicly. 

```bash
python csd_train.py --data_dir LOCATION_OF_DATA --nEpochs 800 --model_filename edsr_csd --neg_num 2 --rgb_range 1 --teacher_model output/model/TEACHER_MODEL_NAME.ckpt
```

## Test

```bash
python test.py --data_dir LOCATION_OF_DATA --model_filename MODEL_NAME.ckpt --pretrained_path ./output/model --data_test Set5 --rgb_range 1 
```