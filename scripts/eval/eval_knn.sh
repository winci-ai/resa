cd ..
port=10001
gpu=0
name=""
pretrained="out/${name}/checkpoint.pth.tar"
bs=256
arch=vit_small  
env_name="eval_knn_${name}"
dump_path="out/${env_name}"

CUDA_VISIBLE_DEVICES=${gpu} torchrun --nproc_per_node=1 --master_port=${port} eval_knn.py \
--batch_size ${bs} \
--arch ${arch} \
--pretrained ${pretrained} \
--data_path ./data/ImageNet/ \