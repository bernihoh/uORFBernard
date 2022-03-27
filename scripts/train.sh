#!/bin/bash
TRAIN_DATA=${1:-'./room_diverse_train'} #Eigentlich room_diverse_train
TEST_DATA=${2:-'./room_diverse_test'}

python train.py --train_dataroot $TRAIN_DATA  --test_dataroot $TEST_DATA \
    --n_scenes 50 --n_img_each_scene 4 --display_grad \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 64 --coarse_epoch 120 \
    --no_locality_epoch 60 --z_dim 64 --num_slots 5 --bottom \
    --batch_size 1 --num_threads 10 --gan_train_epoch 1 --checkpoint 'checkpoints' --version 'original'\

# done
echo "Done"