#!/bin/bash

port=10001
gpu=0
lr_classifier=40
lr_encoder=0.0002
epochs=20
name=""
pretrained="out/${name}/checkpoint.pth.tar"
bs=256
arch=resnet50
train_percent=1  ## or 10 

CUDA_VISIBLE_DEVICES=${gpu} torchrun --nproc_per_node=1 --master_port=${port} eval_linear.py \
--epochs ${epochs} \
--arch ${arch} \
--lr_classifier ${lr_classifier} \
--lr_encoder ${lr_encoder} \
--weights finetune \
--train_percent ${train_percent} \
--batch_size ${bs} \
--pretrained ${pretrained} \
--data_path /path/to/imagenet \
--dump_path /path/to/saving_dir \