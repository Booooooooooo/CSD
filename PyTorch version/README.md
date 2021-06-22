## Dependencies

PyTorch

matplotlib

imageio

tensorboardX

opencv-python 

scipy

scikit-image

## Train

### Prepare data

We use DIV2K training set as our training data. 

About how to download data, you could refer to [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch)

### Begin to train

```python
# use '--teacher_model' to specify the teacher model
python main.py --model EDSR --scale 4 --reset --dir_data LOCATION_OF_DATA --model_filename edsr_x4_0.25student --pre_train output/model/edsr/ --epochs 400 --model_stat --neg_num 10 --contra_lambda 200 --t_lambda 1 --t_l_remove 400 --contrast_t_detach 
```

## Test

You could download models of our paper from [BaiduYun](https://pan.baidu.com/s/1gYenkfLac1s19lfzczxtHA )(code: zser).

1. Test on benchmarks:

``` 
python main.py --scale 4 --pre_train FOLDER_OF_THE_PRETRAINED_MODEL --model_filename edsr_x4_0.25student --test_only --self_ensemble --dir_demo test --model EDSR --dir_data LOCATION_OF_DATA --n_GPUs 1 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --stu_width_mult 0.25 --model_stat --data_test Set5
```

2. Test your own image:

   ```python
   python main.py --scale 4 --pre_train output/model --model_filename edsr_x4_0.25student --test_only --self_ensemble --model EDSR --n_GPUs 1 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --stu_width_mult 0.25 --model_stat --data_test Demo --save_results --dir_demo LOCATION_OF_YOUR_IMAGE 
   ```

   The output SR image will be save in './test/result/'

## Citation 

If you find the code helpful in you research or work, please cite as:

## Acknowledgements

This code is built on [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch). We thank the authors for sharing their codes. 

