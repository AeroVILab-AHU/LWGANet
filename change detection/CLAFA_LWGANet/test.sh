CUDA_VISIBLE_DEVICES=0 python test.py --dataset LEVIR_256_split --checkpoint_dir ./checkpoints/LEVIR_e200

CUDA_VISIBLE_DEVICES=0 python test.py --dataset WHU_256         --checkpoint_dir ./checkpoints/WHU_256_e200

CUDA_VISIBLE_DEVICES=0 python test.py --dataset CDD_256         --checkpoint_dir ./checkpoints/CDD_256_e200

CUDA_VISIBLE_DEVICES=0 python test.py --dataset SYSU_256        --checkpoint_dir ./checkpoints/SYSU_256_e200
