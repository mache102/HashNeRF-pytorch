# CUDA_VISIBLE_DEVICES=0 python run_nerf.py --config configs/chair.txt --finest_res 1024
# CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/chair.txt --finest_res 1024 --em_views 0
# CUDA_VISIBLE_DEVICES=2 python run_nerf.py --config configs/chair.txt --finest_res 1024 --lr 0.01 --lr_decay 100
# CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/chair.txt --finest_res 1024 --log2_hashmap_size 14 

# CUDA_VISIBLE_DEVICES=0 python run_nerf.py --config configs/chair.txt --finest_res 1024 
# CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/chair.txt --finest_res 1024 --lr 0.01 --lr_decay 100
# CUDA_VISIBLE_DEVICES=2 python run_nerf.py --config configs/chair.txt --finest_res 1024 --em_xyz 0 --em_views 0
# CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/chair.txt --finest_res 1024 --em_xyz 0 --em_views 0 --lr 0.01 --lr_decay 100

python3 main.py --config configs/gmap_rail1.txt --embed_config hash_encoding --model_config hash_nerf --train_iters 10000 --lr 0.01 --lr_decay 10