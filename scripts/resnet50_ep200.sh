port=10001
gpu=0,1,2,3
lr=0.5
wd=1e-5
optimizer=sgd
epochs=100 ## change here for 200, 800 epochs pre-training
warmup=2
arch=resnet50
bs=256
gpu_list=(${gpu//,/ })
ngpu=${#gpu_list[@]}
bs_total=$((bs * ngpu))
env_name="resa_${arch}_ep${epochs}_bs${bs_total}_optim${optimizer}_lr${lr}_wd${wd}_warmup${warmup}"
dump_path="out/${env_name}"

CUDA_VISIBLE_DEVICES=${gpu} torchrun --nproc_per_node=${ngpu} --master_port=${port} main.py \
--arch ${arch} \
--epochs ${epochs} \
--optimizer ${optimizer} \
--batch_size ${bs} \
--lr ${lr} \
--wd ${wd} \
--warmup_epochs ${warmup} \
--data_path ./data/ImageNet/ \
--dump_path ${dump_path} \
--workers 8 \