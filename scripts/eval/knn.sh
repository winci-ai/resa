cd ..
port=10001
gpu=0
name=""
pretrained="out/${name}/checkpoint.pth.tar"
bs=256
arch=vit_small
patch_size=16

CUDA_VISIBLE_DEVICES=${gpu} torchrun --nproc_per_node=1 --master_port=${port} eval_knn.py \
--arch ${arch} \
--patch_size ${patch_size} \
--batch_size ${bs} \
--pretrained ${pretrained} \
--data_path /path/to/imagenet \
--dump_path /path/to/saving_dir \