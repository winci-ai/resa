#!/bin/bash

port=10001
gpu=0,1,2,3
epochs=100  # change here for 200, 800 epochs pre-training
arch=resnet50
bs=256
gpu_list=(${gpu//,/ })
ngpu=${#gpu_list[@]}
bs_total=$((bs * ngpu))
env_name="resa_${arch}_ep${epochs}_bs${bs_total}"
dump_path="out/${env_name}"

CUDA_VISIBLE_DEVICES=${gpu} torchrun --nproc_per_node=${ngpu} --master_port=${port} main.py \
--arch ${arch} \
--epochs ${epochs} \
--batch_size ${bs} \
--data_path ./data/ImageNet/ \
--dump_path ${dump_path} \