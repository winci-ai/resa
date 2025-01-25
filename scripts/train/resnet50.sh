#!/bin/bash

port=10001
gpu=0,1,2,3
epochs=100  # change here for 200 epochs pre-training
arch=resnet50
bs=256

CUDA_VISIBLE_DEVICES=${gpu} torchrun --nproc_per_node=4 --master_port=${port} main.py \
--arch ${arch} \
--epochs ${epochs} \
--batch_size ${bs} \
--data_path /path/to/imagenet \
--dump_path /path/to/saving_dir \