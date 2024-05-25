CUDA_VISIBLE_DEVICES=1 python train_supervision.py -c config/loveda/unetformer.py
CUDA_VISIBLE_DEVICES=1 python loveda_test.py -c config/loveda/unetformer.py -o fig_results/loveda_e100/unetformer -t 'd4'
