#!/bin/bash

port=10001
gpu=0,1,2,3
epochs=300
arch=vit_small
patch_size=16
bs=256

CUDA_VISIBLE_DEVICES=${gpu} torchrun --nproc_per_node=${ngpu} --master_port=${port} main.py \
--arch ${arch} \
--patch_size ${patch_size} \
--epochs ${epochs} \
--batch_size ${bs} \
--data_path /path/to/imagenet \
--dump_path /path/to/saving_dir \