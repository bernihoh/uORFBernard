#!/bin/bash
DATAROOT=${1:-'./room_diverse_test'}
CHECKPOINT=${2:-'./lightning_logs/Final_4gpu/sa2d1/version_20499894/checkpoints/epoch=240-step=301249.ckpt'}

python predict.py --train_dataroot "" --test_dataroot $DATAROOT --n_scenes 3 --n_img_each_scene 4 \
    --checkpoint $CHECKPOINT \
    --load_size 128 --input_size 128 --render_size 8 --frustum_size 128 --bottom \
    --n_samp 256 --z_dim 64 --num_slots 5 --gpus 1 --version "sa2d1"\

echo "Done"