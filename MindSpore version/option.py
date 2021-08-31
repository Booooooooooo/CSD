import argparse
parser = argparse.ArgumentParser(description='Mindspore Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=3, help='Snapshots')  # 25
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
# parser.add_argument('--data_dir', type=str, default='/home/hoo/ms_dataset/DIV2K')
parser.add_argument('--data_dir', type=str, default='D://dataset/')
parser.add_argument('--data_train', type=str, default='DIV2K', help='training dataset')
parser.add_argument('--data_test', type=str, default='DIV2K', help='testing dataset')
parser.add_argument('--data_range', type=str, default='1-800/801-810',help='train/test data range')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--hr_train_dataset', type=str, default='DIV2K_train_HR')
# parser.add_argument('--hr_train_dataset', type=str, default='DIV2KDemo')
parser.add_argument('--model_type', type=str, default='DBPNLL')
parser.add_argument('--patch_size', type=int, default=60, help='Size of cropped HR image')
parser.add_argument('--feature_extractor', default='VGG', help='Location to save checkpoint models')
parser.add_argument('--w1', type=float, default=1e-2, help='MSE weight')
parser.add_argument('--w2', type=float, default=1e-1, help='Perceptual weight')
parser.add_argument('--w3', type=float, default=1e-3, help='Adversarial weight')
parser.add_argument('--w4', type=float, default=10, help='Style weight')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='PIRM_VGG', help='Location to save checkpoint models')
parser.add_argument('--pretrained_path', default='./output/model', help='Location to save checkpoint models')
parser.add_argument('--model', type=str, default='edsr', help='model type')

##model args
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--n_resblocks', type=int, default=32,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=256,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=0.1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--scale', type=str, default='4',
                    help='super resolution scale')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--contra_lambda', type=float, default=1, help='weight of contra_loss')

parser.add_argument('--stu_width_mult', type=float, default=0.25,
                    help='width_mult of student model')
parser.add_argument('--model_filename', type=str, default='', help='pre-train model filename')
parser.add_argument('--neg_num', type=int, default=10, help='negative samples number')
parser.add_argument('--teacher_model', type=str, default='output/model/edsr_baseline_meanshift-800_50.ckpt',
                    help='load teacher model from {teacher_model}')

# ModelArts参数
parser.add_argument('--data_url', type=str, default='')
parser.add_argument('--train_url', type=str, default='')

opt = parser.parse_args()
opt.scale = list(map(lambda x: int(x), opt.scale.split('+')))
# gpus_list = range(opt.gpus)
# opt.data_dir = opt.data_url
print(opt)
