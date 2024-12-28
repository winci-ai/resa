#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --output=%j.log   
#SBATCH --error=%j.log     
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --job-name=resa_vit_pretrain
#SBATCH --mem=450G

conda activate resa
cd ..

lr=5e-4
wd=0.1
optimizer=adamw
epochs=300
warmup=40
arch=vit_small
patch_size=16
bs=256
bs_total=$((bs * 4))
wandb=resa_vit_imagenet
env_name="resa_${arch}_ep${epochs}_bs${bs_total}_${optimizer}_lr${lr}_wd${wd}_warmup${warmup}"
dump_path="out/${env_name}"

MASTER_PORT=29500
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) # 主节点地址
echo "MASTER_ADDR is ${MASTER_ADDR}"
echo "MASTER_PORT is ${MASTER_PORT}"

srun --output=${dump_path}/%j.out --error=${dump_path}/%j.err --label \
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=$SLURM_JOB_ID --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} train.py \
    --arch ${arch} \
    --patch_size ${patch_size} \
    --crops_nmb 1 \
    --crops_size 224 \
    --solarization_prob 0.2 \
    --epochs ${epochs} \
    --env_name ${env_name} \
    --wandb ${wandb} \
    --optimizer ${optimizer} \
    --batch_size ${bs} \
    --lr ${lr} \
    --wd ${wd} \
    --temperature 0.2 \
    --warmup_epochs ${warmup} \
    --data_path ./data/ImageNet/ \
    --dump_path ${dump_path} \
    --workers 8 \