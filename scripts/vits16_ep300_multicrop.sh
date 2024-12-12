port=10001
gpu=0,1,2,3,4,5,6,7
lr=5e-4
wd=0.1
optimizer=adamw
epochs=300
warmup=40
arch=vit_small
patch_size=16
bs=128
gpu_list=(${gpu//,/ })
ngpu=${#gpu_list[@]}
bs_total=$((bs * ngpu))
env_name="resa_multicrop_${arch}_ep${epochs}_bs${bs_total}_optim${optimizer}_lr${lr}_wd${wd}_warmup${warmup}"
dump_path="out/${env_name}"

CUDA_VISIBLE_DEVICES=${gpu} torchrun --nproc_per_node=${ngpu} --master_port=${port} main.py \
--arch ${arch} \
--patch_size ${patch_size} \
--nmb_crops 1 6 \
--crops_size 224 96 \
--min_scale_crops 0.25 0.05 \
--max_scale_crops 1 0.25 \
--gaussian_prob 0.5 0.5 \
--solarization_prob 0.2 0 \
--epochs ${epochs} \
--optimizer ${optimizer} \
--batch_size ${bs} \
--lr ${lr} \
--wd ${wd} \
--warmup_epochs ${warmup} \
--data_path ./data/ImageNet/ \
--dump_path ${dump_path} \
--workers 8 \