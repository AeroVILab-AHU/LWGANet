CUDA_VISIBLE_DEVICES=1 python train_supervision.py -c config/loveda/unetformer_lwganet_l2_e30.py
CUDA_VISIBLE_DEVICES=1 python loveda_test.py -c config/loveda/unetformer_lwganet_l2_e30.py -o fig_results/loveda/unetformer_lwganet_l2_e30 -t 'd4'

