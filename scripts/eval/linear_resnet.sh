#!/bin/bash

port=10001
gpu=0
lr=40  ## change lr to 10 when pre-training bs=256
epochs=100
name=""
pretrained="out/${name}/checkpoint.pth.tar"
bs=256
arch=resnet50

CUDA_VISIBLE_DEVICES=${gpu} torchrun --nproc_per_node=1 --master_port=${port} eval_linear.py \
--epochs ${epochs} \
--arch ${arch} \
--lr_classifier ${lr} \
--batch_size ${bs} \
--pretrained ${pretrained} \
--data_path /path/to/imagenet \
--dump_path /path/to/saving_dir \