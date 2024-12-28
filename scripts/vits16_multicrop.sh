#!/bin/bash

port=10001
gpu=0,1,2,3,4,5,6,7
epochs=300
arch=vit_small
patch_size=16
bs=128
gpu_list=(${gpu//,/ })
ngpu=${#gpu_list[@]}
bs_total=$((bs * ngpu))
env_name="resa_multicrop_${arch}-${patch_size}_ep${epochs}_bs${bs_total}"
dump_path="out/${env_name}"

CUDA_VISIBLE_DEVICES=${gpu} torchrun --nproc_per_node=${ngpu} --master_port=${port} main.py \
--arch ${arch} \
--patch_size ${patch_size} \
--crops_nmb 1 6 \
--crops_size 224 96 \
--crops_min_scale 0.25 0.05 \
--crops_max_scale 1.0 0.25 \
--solarization_prob 0.2 0 \
--epochs ${epochs} \
--batch_size ${bs} \
--data_path ./data/ImageNet/ \
--dump_path ${dump_path} \