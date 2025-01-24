#!/bin/bash

port=10001
gpu=0,1,2,3
epochs=300
arch=vit_small
patch_size=16
bs=256
gpu_list=(${gpu//,/ })
ngpu=${#gpu_list[@]}
bs_total=$((bs * ngpu))
dump_path="out/resa_${arch}-${patch_size}_ep${epochs}_bs${bs_total}"

CUDA_VISIBLE_DEVICES=${gpu} torchrun --nproc_per_node=${ngpu} --master_port=${port} main.py \
--arch ${arch} \
--patch_size ${patch_size} \
--epochs ${epochs} \
--batch_size ${bs} \
--data_path ./data/ImageNet/ \
--dump_path ${dump_path} \