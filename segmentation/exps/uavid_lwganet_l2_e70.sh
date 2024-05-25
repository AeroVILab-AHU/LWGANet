CUDA_VISIBLE_DEVICES=2 python ../train_supervision.py -c ../config/uavid/unetformer_lwganet_l2_e70.py
CUDA_VISIBLE_DEVICES=2 python ../inference_uavid.py -c ../config/uavid/unetformer_lwganet_l2_e70.py -o ../fig_results/uavid/lwganet_l2_e70 -t 'lr' -ph 1024 -pw 1024 -b 2 -d "uavid"
