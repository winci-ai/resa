<h2 align="center"> <a href="https://arxiv.org/abs/2501.18452">ReSA: positive-feedback self-supervised learning</a><h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2501.18452-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2501.18452)[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/winci-ai/resa/blob/main/LICENSE)


<p align="center">
    <img src="https://github.com/winci-ai/resa/releases/download/figure/postive-feedback.jpg" width="350">
</p>


## Method
ReSA is designed within our positive-feedback SSL framework, which directly leverages the stable and semantically meaningful clustering properties of the encoder’s outputs. Through an online self-clustering mechanism, ReSA refines its own
objective during training, outperforming state of-the-art SSL baselines and achieving higher efficiency.

<p align="center">
    <img src="https://github.com/winci-ai/resa/releases/download/figure/resa.jpg" width="600">
</p>

### ReSA Excels at both Fine-grained and Coarse-grained Learning

<p align="center">
    <img src="https://github.com/winci-ai/resa/releases/download/figure/tsne.jpg" width="800">
</p>

## Pretrained models

You can choose to download only the weights of the pretrained encoder used for downstream tasks, or the full checkpoint which contains encoder, projector, and predictor weights for both base and momentum networks.

ReSA pretrained ResNet-50 models on ImageNet with two $224 \times 224$ augmented views:

<table align="center" border="1" style="width:100%; border-collapse:collapse; text-align:center;">
  <tr>
    <th>epochs</th>
    <th>bs</th>
    <th>linear acc</th>
    <th>knn acc</th>
    <th colspan="4">download</th>
  </tr>
  <tr>
    <td>100</td>
    <td>256</td>
    <td>71.9%</td>
    <td>64.6%</td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_only_bs256_ep100.pth">ResNet-50</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_full_bs256_ep100.pth">full checkpoint</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_bs256_ep100_train.log">train log</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_bs256_ep100_linear.log">eval log</a></td>
  </tr>
  <tr>
    <td>200</td>
    <td>256</td>
    <td>73.4%</td>
    <td>67.1%</td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_only_bs256_ep200.pth">ResNet-50</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_full_bs256_ep200.pth">full checkpoint</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_bs256_ep200_train.log">train log</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_bs256_ep200_linear.log">eval log</a></td>
  </tr>
  <tr>
    <td>100</td>
    <td>1024</td>
    <td>71.3%</td>
    <td>63.3%</td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_only_bs1024_ep100.pth">ResNet-50</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_full_bs1024_ep100.pth">full checkpoint</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_bs1024_ep100_train.log">train log</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_bs1024_ep100_linear.log">eval log</a></td>
  </tr>
  <tr>
    <td>200</td>
    <td>1024</td>
    <td>73.8%</td>
    <td>67.6%</td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_only_bs1024_ep200.pth">ResNet-50</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_full_bs1024_ep200.pth">full checkpoint</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_bs1024_ep200_train.log">train log</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_bs1024_ep200_linear.log">eval log</a></td>
  </tr>
  <tr>
    <td>800</td>
    <td>1024</td>
    <td>75.2%</td>
    <td>69.9%</td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_only_bs1024_ep800.pth">ResNet-50</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_full_bs1024_ep800.pth">full checkpoint</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_bs1024_ep800_train.log">train log</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_bs1024_ep800_linear.log">eval log</a></td>
  </tr>
</table>

## Code Structure

```
.
├── method                      #   self-supervised method
│   ├── base.py                 #   base class for self-supervised loss implementation
│   └── resa.py                 #   Implementation of ReSA
├── src                         #   the packages
│   ├── imagenet_subset         #   1% and 10% subsets of ImageNet-1K
│   ├── model.py                #   function definition for the encoder, projector, and predictor
│   ├── resnet.py               #   class definition for the ResNet model
│   ├── transforms.py           #   data augmentation for pretraining
│   └── utils.py                #   shared utilities
├── args.py                     #   arguments
├── eval_knn.py                 #   evaluate with a weighted k-nn classifier
├── eval_linear.py              #   evaluate with a linear classifier
└── main.py                     #   main function to pretrain with ReSA
```

## Installation and Requirements

1. Clone this repository and navigate to the folder
```bash
git clone https://github.com/winci-ai/resa.git
cd resa
```

2. Create a conda environment, activate it and install packages (newer versions of python are supported)
```Shell
conda create -n resa python=3.8.18
conda activate resa
pip install -r requirements.txt
```

## Pretraining on ImageNet

If you would like to pretrain ReSA on CIFAR-10/100, please refer to another [repository](https://github.com/winci-ai/resa_cifar).

### ResNet-50 with 1-node (1-GPU) training, a batch size of 256 

```
torchrun --nproc_per_node=1 main.py \
--epochs 100 \
--data_path /path/to/imagenet \
--dump_path /path/to/saving_dir \
```

This process requires approximately 25.2GB of GPU memory, making it well-suited for training on a single V100 GPU. This pretrained model will achieve 71.9% top-1 accuracy with a linear classifier. However, if training is to be conducted on two 3090 or 4090 GPUs, it should be implemented as `--nproc_per_node=2` and `--batch_size 128`.


The command for 200-epoch pretraining is identical; you simply need to set `--epoch 200`.

### ResNet-50 with 1-node (4-GPU) training, a batch size of 1024

```
torchrun --nproc_per_node=4 main.py \
--epochs 100 \
--warmup_epochs 10 \
--data_path /path/to/imagenet \
--dump_path /path/to/saving_dir \
```

When pretraining for 800 epochs, you should set an extra `--lr 0.4` to ensure training stability.

## Evaluation

### Linear classification

If the pretraining batch size is 1024, just run:
```
torchrun --nproc_per_node=1 eval_linear.py \
--epochs 100 \
--data_path /path/to/imagenet \
--dump_path /path/to/saving_dir \
--pretrained /path/to/checkpoint.pth \
```

If the pretraining batch size is 256, you should set an extra `--lr_classifier 10`.

### K-NN classification

To evaluate a weighted k-NN classifier with a single GPU on a pre-trained model, run:

```
torchrun --nproc_per_node=1 eval_knn.py \
--data_path /path/to/imagenet \
--dump_path /path/to/saving_dir \
--pretrained /path/to/checkpoint.pth \
```

### Semi-supervised classification

To perform a low-shot evaluation on the ResNet-50 model pretrained for 800 epochs with a batch size of 1024, run
```
torchrun --nproc_per_node=1 eval_linear.py \
--epochs 20 \
--scheduler cos \
--weights finetune \
--train_percent 1 \
--data_path /path/to/imagenet \
--dump_path /path/to/saving_dir \
--pretrained /path/to/checkpoint.pth \
```

The command for using 10% subset is identical; you simply need to set `--train_percent 10`.


## License
See the [LICENSE](./LICENSE) file for details about the license under which this code is made available.

## Citation
If you find this repository useful in your research, please consider giving a star :star: and a citation
```
@article{weng2025resa,
  title={Clustering Properties of Self-Supervised Learning},
  author={Weng, Xi and An, Jianing and Ma, Xudong and Qi, Binhang and Luo, Jie and Yang, Xi and Dong, Jin Song and Huang, Lei},
  journal={arXiv preprint arXiv:2501.18452},
  year={2025}
}
