
CUDA_VISIBLE_DEVICES=0 python main.py --dataset LEVIR_256_split --checkpoint_dir ./checkpoints/lwganet_l0/LEVIR_e200

CUDA_VISIBLE_DEVICES=0 python main.py --dataset WHU_256         --checkpoint_dir ./checkpoints/lwganet_l0/WHU_256_e200

CUDA_VISIBLE_DEVICES=0 python main.py --dataset CDD_256         --checkpoint_dir ./checkpoints/lwganet_l0/CDD_256_e200

CUDA_VISIBLE_DEVICES=0 python main.py --dataset SYSU_256        --checkpoint_dir ./checkpoints/lwganet_l0/SYSU_256_e200
