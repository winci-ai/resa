#!/bin/bash

port=10001
gpu=0 
lr=0.03  ## change lr to 8 when pre-training bs=256
epochs=100
name=""
pretrained="out/${name}/checkpoint.pth.tar"
bs=256
arch=vit_small
patch_size=16
env_name="eval_lr${lr}_${name}"
dump_path="out/${env_name}"

CUDA_VISIBLE_DEVICES=${gpu} torchrun --nproc_per_node=1 --master_port=${port} eval_linear.py \
--epochs ${epochs} \
--arch ${arch} \
--patch_size ${patch_size} \
--lr-classifier ${lr} \
--batch_size ${bs} \
--dump_path ${dump_path} \
--pretrained ${pretrained} \
--data_path ./data/ImageNet/ \