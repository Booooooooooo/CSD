### train  feature affinity based distillation loss
CUDA_VISIBLE_DEVICES=1 nohup python main.py --model EDSR --scale 4 --reset --dir_data /data/dataset/ --model_filename edsr_x4_0.25student_faloss --pre_train output/model/edsr/ --epochs 400 --model_stat --neg_num 10 --contra_lambda 200 --t_lambda 1 --t_l_remove 400 --contrast_t_detach > output/edsr_x4_0.25faloss.out 2>&1 &

### rcan [0.75, 1.0]
CUDA_VISIBLE_DEVICES=2 nohup python main.py --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --scale 4 --reset --dir_data /home/ubuntu/data/hdd1/wyb_dataset --model_filename rcan_x4_0.75student --pre_train output/model/rcan/ --epochs 400 --model_stat --neg_num 10 --contra_lambda 200 --t_lambda 1 --t_l_remove 400 --contrast_t_detach --teacher_model output/model/rcan/baseline/rcan_x4_baseline.pth --stu_width_mult 0.75 > output/rcan_x4_075student.out 2>&1 &
## edsr [0.75, 1.0]
nohup python main.py --model EDSR --scale 4 --reset --dir_data /data/dataset/ --model_filename edsr_x4_0.75student --pre_train output/model/rcan/ --epochs 400 --model_stat --neg_num 10 --contra_lambda 200 --t_lambda 1 --t_l_remove 400 --contrast_t_detach --stu_width_mult 0.75 > output/edsr_x4_0.75student.out 2>&1 &



#test
CUDA_VISIBLE_DEVICES=1 python main.py --scale 4 --pre_train output/model --model_filename edsr_x4_0.25student --test_only --self_ensemble --dir_demo test --model EDSR --dir_data /home/ubuntu/data/hdd1/wyb_dataset --n_GPUs 1 --n_resblocks 32 --n_feats 256 --stu_width_mult 0.25 --model_stat --data_test B100
CUDA_VISIBLE_DEVICES=2 python main.py --scale 4 --pre_train output/model --model_filename rcan_x4_0.75student --test_only --self_ensemble --dir_demo test --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --dir_data /home/ubuntu/data/hdd1/wyb_dataset --stu_width_mult 0.75 --model_stat --data_test Urban100
